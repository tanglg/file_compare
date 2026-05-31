import { invoke } from "@tauri-apps/api/core";
import { open } from "@tauri-apps/plugin-dialog";
import {
  AlertTriangle,
  Check,
  Download,
  FolderOpen,
  Loader2,
  Play,
  Plus,
  Settings2,
  Square,
  X,
} from "lucide-react";
import { useEffect, useMemo, useState } from "react";

type AnalysisStage =
  | "Init"
  | "ReadingMeta"
  | "BuildingTextIndex"
  | "RecallingCandidates"
  | "ComparingText"
  | "GeneratingReport"
  | "Done"
  | "Cancelled"
  | "Failed";

type AnalysisProgress = {
  task_id: string;
  stage: AnalysisStage;
  current_file?: string | null;
  current_page?: number | null;
  processed_files: number;
  total_files: number;
  processed_pages: number;
  total_pages: number;
  indexed_chunks: number;
  indexed_images: number;
  cache_hits: number;
  candidate_pairs: number;
  confirmed_pairs: number;
  similarity_groups: number;
  weak_connection_groups: number;
  confirmed_text_matches: number;
  confirmed_image_matches: number;
  elapsed_seconds: number;
  estimated_remaining_seconds?: number | null;
  message: string;
};

type FileSummary = {
  id: string;
  path: string;
  file_name: string;
  page_count: number;
  total_text_chars: number;
  chunk_count: number;
  image_count: number;
  indexed_image_count: number;
  status: string;
  error?: string | null;
};

type MatchedText = {
  left_page: number;
  right_page: number;
  similarity: number;
  text_readable: boolean;
  left_text: string;
  right_text: string;
};

type MatchedImage = {
  left_page: number;
  right_page: number;
  hamming_distance: number;
  similarity: number;
  exact: boolean;
  width: number;
  height: number;
};

type SimilarityPair = {
  pair_id: string;
  left_file_id: string;
  right_file_id: string;
  left_file: string;
  right_file: string;
  text_score: number;
  image_score: number;
  page_image_score: number;
  final_score: number;
  level: SimilarityLevel;
  exact_page_match_count: number;
  approximate_text_match_count: number;
  matched_text_chars: number;
  matched_texts: MatchedText[];
  matched_images: MatchedImage[];
};

type PairRelation = {
  left_file_id: string;
  right_file_id: string;
  left_file: string;
  right_file: string;
  final_score: number;
  text_score: number;
  image_score: number;
  page_image_score: number;
};

type SimilarityGroup = {
  group_id: string;
  file_ids: string[];
  files: string[];
  group_score: number;
  level: SimilarityLevel;
  graph_density: number;
  quality_flags: GroupQualityFlag[];
  pair_relations: PairRelation[];
};

type AnalysisResult = {
  task_id: string;
  started_at: string;
  finished_at?: string | null;
  files: FileSummary[];
  pairs: SimilarityPair[];
  groups: SimilarityGroup[];
  report_path?: string | null;
  warnings: string[];
};

type AnalysisConfig = {
  analysis_depth: string;
  text_threshold: number;
  image_threshold: number;
  final_threshold: number;
  min_chunk_chars: number;
  target_chunk_chars: number;
  chunk_overlap_chars: number;
  shingle_size: number;
  min_shared_shingles: number;
  simhash_hamming_threshold: number;
  candidate_score_threshold: number;
  candidate_top_k_per_file: number;
  max_matches_per_pair: number;
};

type AnalyzeRequest = AnalysisConfig & { paths: string[] };

type ExportReportResult = {
  exported_files: string[];
};

type SimilarityLevel = "Extreme" | "High" | "Medium" | "Low";
type GroupQualityFlag = "WeakConnection" | "NeedsManualReview";

const analysisPresets = {
  fast: {
    analysis_depth: "fast",
    text_threshold: 0.78,
    image_threshold: 0.8,
    final_threshold: 0.56,
    min_chunk_chars: 120,
    target_chunk_chars: 620,
    chunk_overlap_chars: 70,
    shingle_size: 6,
    min_shared_shingles: 4,
    simhash_hamming_threshold: 3,
    candidate_score_threshold: 0.42,
    candidate_top_k_per_file: 12,
    max_matches_per_pair: 24,
  },
  balanced: {
    analysis_depth: "balanced",
    text_threshold: 0.72,
    image_threshold: 0.8,
    final_threshold: 0.5,
    min_chunk_chars: 80,
    target_chunk_chars: 500,
    chunk_overlap_chars: 80,
    shingle_size: 5,
    min_shared_shingles: 3,
    simhash_hamming_threshold: 4,
    candidate_score_threshold: 0.35,
    candidate_top_k_per_file: 20,
    max_matches_per_pair: 30,
  },
  deep: {
    analysis_depth: "deep",
    text_threshold: 0.68,
    image_threshold: 0.8,
    final_threshold: 0.45,
    min_chunk_chars: 60,
    target_chunk_chars: 420,
    chunk_overlap_chars: 120,
    shingle_size: 4,
    min_shared_shingles: 2,
    simhash_hamming_threshold: 6,
    candidate_score_threshold: 0.24,
    candidate_top_k_per_file: 36,
    max_matches_per_pair: 60,
  },
  exhaustive: {
    analysis_depth: "exhaustive",
    text_threshold: 0.62,
    image_threshold: 0.78,
    final_threshold: 0.4,
    min_chunk_chars: 40,
    target_chunk_chars: 320,
    chunk_overlap_chars: 160,
    shingle_size: 3,
    min_shared_shingles: 1,
    simhash_hamming_threshold: 8,
    candidate_score_threshold: 0.16,
    candidate_top_k_per_file: 64,
    max_matches_per_pair: 100,
  },
} satisfies Record<string, AnalysisConfig>;

type AnalysisPreset = keyof typeof analysisPresets;

const analysisPresetDetails: Record<
  AnalysisPreset,
  { label: string; tag: string; description: string; recommendation: string }
> = {
  fast: {
    label: "快速",
    tag: "初步筛查",
    description: "减少候选数量，优先确认明显重复内容。",
    recommendation: "适合文件较多、先快速摸底；可能略过弱关联或局部改写。",
  },
  balanced: {
    label: "均衡",
    tag: "默认推荐",
    description: "在召回范围、证据数量和执行耗时之间取平衡。",
    recommendation: "适合日常检测，通常先用这一档。",
  },
  deep: {
    label: "深度",
    tag: "加强召回",
    description: "切块更细、重叠更多，并保留更多候选和证据。",
    recommendation: "适合复核重点批次，耗时会明显增加。",
  },
  exhaustive: {
    label: "穷尽",
    tag: "最深检查",
    description: "进一步放宽召回条件，尽量发现零散复用和较弱相似。",
    recommendation: "适合最终审查；耗时最高，也会产生更多需人工确认的弱候选。",
  },
};

const stageOrder: AnalysisStage[] = [
  "ReadingMeta",
  "BuildingTextIndex",
  "RecallingCandidates",
  "ComparingText",
  "GeneratingReport",
];

const stageLabels: Record<AnalysisStage, string> = {
  Init: "等待任务",
  ReadingMeta: "快速扫描",
  BuildingTextIndex: "全局索引",
  RecallingCandidates: "候选召回",
  ComparingText: "候选精算",
  GeneratingReport: "图聚类",
  Done: "分析完成",
  Cancelled: "任务已取消",
  Failed: "任务失败",
};

const pipeline = [
  ["快速扫描", "逐页抽取文本与内嵌图片"],
  ["全局索引", "文本 shingle 与图片指纹"],
  ["候选召回", "文本、图片候选合并与降噪"],
  ["候选精算", "文本覆盖与图片一对一匹配"],
  ["图聚类", "输出雷同组"],
];

function basename(path: string) {
  return path.split(/[\\/]/).pop() ?? path;
}

function compactName(name: string, max = 13) {
  const plain = name.replace(/\.pdf$/i, "");
  return plain.length > max ? `${plain.slice(0, max)}…` : plain;
}

function score(value: number) {
  return `${Math.round(value * 100)}%`;
}

function number(value: number) {
  return new Intl.NumberFormat("zh-CN").format(value);
}

function chars(value: number) {
  return value >= 10_000 ? `${(value / 10_000).toFixed(value >= 100_000 ? 0 : 1)}万` : number(value);
}

function percent(progress?: AnalysisProgress | null) {
  if (!progress) return 0;
  if (progress.stage === "Done") return 100;
  const total = Math.max(progress.total_pages, 1);
  const pageProgress = progress.processed_pages / total;
  const stageBonus =
    progress.stage === "BuildingTextIndex"
      ? 0.08
      : progress.stage === "RecallingCandidates"
        ? 0.11
        : progress.stage === "ComparingText"
          ? 0.14
          : progress.stage === "GeneratingReport"
            ? 0.18
            : 0;
  return Math.min(99, Math.round(pageProgress * 82 + stageBonus * 100));
}

function levelLabel(level: SimilarityLevel) {
  return (
    {
      Extreme: "极高",
      High: "高",
      Medium: "中",
      Low: "低",
    }[level] ?? level
  );
}

function statusLabel(status: string) {
  return (
    {
      ready: "已索引",
      "cid-fallback": "CID 指纹",
      "image-only": "仅图片",
      failed: "失败",
      "text-empty": "文本较少",
    }[status] ?? status
  );
}

function activePipelineStep(progress?: AnalysisProgress | null) {
  if (!progress) return 0;
  if (progress.stage === "Done") return 5;
  if (progress.stage === "ReadingMeta") return 1;
  if (progress.stage === "BuildingTextIndex") return 2;
  if (progress.stage === "RecallingCandidates") return 3;
  if (progress.stage === "ComparingText") return 4;
  if (progress.stage === "GeneratingReport") return 5;
  return 0;
}

export function App() {
  const [paths, setPaths] = useState<string[]>([]);
  const [taskId, setTaskId] = useState<string | null>(null);
  const [progress, setProgress] = useState<AnalysisProgress | null>(null);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [selectedPairId, setSelectedPairId] = useState<string | null>(null);
  const [selectedGroupId, setSelectedGroupId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [finalThreshold, setFinalThreshold] = useState(0.5);
  const [analysisConfig, setAnalysisConfig] = useState<AnalysisConfig>(analysisPresets.balanced);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [exportWord, setExportWord] = useState(true);
  const [exportJson, setExportJson] = useState(true);
  const [includeTextEvidence, setIncludeTextEvidence] = useState(true);
  const [exportMessage, setExportMessage] = useState<string | null>(null);

  const isRunning = progress ? !["Done", "Cancelled", "Failed"].includes(progress.stage) : false;
  const files = result?.files ?? [];
  const groups = result?.groups ?? [];
  const sortedPairs = useMemo(
    () =>
      [...(result?.pairs ?? [])]
        .filter((pair) => pair.final_score >= finalThreshold)
        .sort((left, right) => right.final_score - left.final_score),
    [finalThreshold, result],
  );
  const selectedGroup = groups.find((group) => group.group_id === selectedGroupId) ?? groups[0];
  const relationPairs = useMemo(
    () =>
      selectedGroup
        ? selectedGroup.pair_relations
            .map((relation) =>
              (result?.pairs ?? []).find(
                (pair) =>
                  pair.left_file_id === relation.left_file_id &&
                  pair.right_file_id === relation.right_file_id,
              ),
            )
            .filter((pair): pair is SimilarityPair => Boolean(pair))
        : [],
    [result, selectedGroup],
  );
  const selectedPair =
    relationPairs.find((pair) => pair.pair_id === selectedPairId) ??
    relationPairs[0] ??
    sortedPairs[0];
  const totalChunks = files.reduce((sum, file) => sum + file.chunk_count, 0);
  const totalImages = files.reduce((sum, file) => sum + file.image_count, 0);
  const indexedImages = files.reduce((sum, file) => sum + file.indexed_image_count, 0);
  const totalEvidence = sortedPairs.reduce(
    (sum, pair) =>
      sum + pair.exact_page_match_count + pair.approximate_text_match_count + pair.matched_images.length,
    0,
  );
  const currentStep = activePipelineStep(progress);

  useEffect(() => {
    if (!taskId || !isRunning) return;
    let disposed = false;
    const timer = window.setInterval(async () => {
      try {
        const next = await invoke<AnalysisProgress>("get_analysis_progress", { taskId });
        if (disposed) return;
        if (["Done", "Cancelled", "Failed"].includes(next.stage)) {
          const finalResult = await invoke<AnalysisResult>("get_analysis_result", { taskId });
          if (disposed) return;
          setProgress(next);
          setResult(finalResult);
          setSelectedGroupId(finalResult.groups[0]?.group_id ?? null);
          setSelectedPairId(finalResult.pairs[0]?.pair_id ?? null);
        } else {
          setProgress(next);
        }
      } catch (cause) {
        if (disposed) return;
        setError(String(cause));
      }
    }, 800);
    return () => {
      disposed = true;
      window.clearInterval(timer);
    };
  }, [taskId, isRunning]);

  async function selectFiles() {
    setError(null);
    const selected = await open({
      multiple: true,
      filters: [{ name: "PDF", extensions: ["pdf"] }],
    });
    if (!selected) return;
    setPaths(Array.isArray(selected) ? selected : [selected]);
    setResult(null);
    setSelectedPairId(null);
    setSelectedGroupId(null);
    setProgress(null);
  }

  async function start() {
    if (paths.length < 2) {
      setError("请至少选择 2 个 PDF 文件。");
      return;
    }
    setError(null);
    setResult(null);
    setSelectedPairId(null);
    setSelectedGroupId(null);
    setExportMessage(null);
    const request: AnalyzeRequest = {
      paths,
      ...analysisConfig,
      final_threshold: finalThreshold,
    };
    try {
      const id = await invoke<string>("create_analysis_task", { request });
      setTaskId(id);
      setProgress(await invoke<AnalysisProgress>("get_analysis_progress", { taskId: id }));
    } catch (cause) {
      setError(String(cause));
    }
  }

  async function cancel() {
    if (!taskId) return;
    try {
      await invoke("cancel_analysis_task", { taskId });
      setProgress(await invoke<AnalysisProgress>("get_analysis_progress", { taskId }));
    } catch (cause) {
      setError(String(cause));
    }
  }

  function applyPreset(profile: keyof typeof analysisPresets) {
    const next = analysisPresets[profile];
    setAnalysisConfig(next);
    setFinalThreshold(next.final_threshold);
  }

  function updateAnalysisConfig(patch: Partial<AnalysisConfig>) {
    setAnalysisConfig((current) => ({ ...current, ...patch, analysis_depth: "custom" }));
    if (patch.final_threshold !== undefined) setFinalThreshold(patch.final_threshold);
  }

  async function exportReport() {
    if (!result) return;
    if (!exportWord && !exportJson) {
      setError("请至少选择一种导出格式。");
      return;
    }
    setError(null);
    setExportMessage(null);
    try {
      const directory = await open({ directory: true, multiple: false, title: "选择报告导出目录" });
      if (!directory) return;
      const exported = await invoke<ExportReportResult>("export_analysis_report", {
        request: {
          task_id: result.task_id,
          target_dir: directory,
          export_json: exportJson,
          export_word: exportWord,
          include_text_evidence: includeTextEvidence,
        },
      });
      setExportMessage(`已导出 ${exported.exported_files.length} 个文件到 ${directory}`);
    } catch (cause) {
      setError(String(cause));
    }
  }

  return (
    <main className="app-shell">
      <header className="topbar">
        <div className="brand">
          <div className="brand-mark">◎</div>
          <div>
            <h1>PDF 雷同性检测工具</h1>
            <p>全局索引 · 候选召回 · 相似图聚类 · 本地隐私保护</p>
          </div>
        </div>
        <div className="top-actions">
          <span className="badge good">文本引擎已启用</span>
          <span className="badge good">权限正常</span>
          <span className="badge blue">本地处理</span>
          <span className="badge good">图片指纹已启用</span>
          <button className="button" onClick={selectFiles}>
            <Plus size={15} />
            导入 PDF
          </button>
          <button className="button primary" onClick={start} disabled={isRunning}>
            {isRunning ? <Loader2 className="spin" size={15} /> : <Play size={15} />}
            {isRunning ? "分析中" : "开始分析"}
          </button>
          <button className="button danger" onClick={cancel} disabled={!isRunning}>
            <Square size={14} />
            取消任务
          </button>
          <button className="button" onClick={() => setSettingsOpen(true)}>
            <Settings2 size={14} />
            参数设置
          </button>
          <button className="button" onClick={exportReport} disabled={!result}>
            <Download size={14} />
            导出报告
          </button>
        </div>
      </header>

      {error && (
        <div className="notice error">
          <AlertTriangle size={16} />
          {error}
        </div>
      )}
      {exportMessage && (
        <div className="notice success">
          <Check size={16} />
          {exportMessage}
        </div>
      )}

      {settingsOpen && (
        <div className="modal-backdrop" role="presentation" onMouseDown={() => setSettingsOpen(false)}>
          <section className="settings-modal" role="dialog" aria-modal="true" onMouseDown={(event) => event.stopPropagation()}>
            <div className="settings-head">
              <div>
                <h2>分析深度参数</h2>
                <p>预设会实际改变候选召回范围、文本切块和证据保留数量。</p>
              </div>
              <button className="icon-button" title="关闭" onClick={() => setSettingsOpen(false)}>
                <X size={17} />
              </button>
            </div>
            <div className="depth-tabs">
              {(Object.keys(analysisPresets) as AnalysisPreset[]).map((profile) => (
                <button
                  className={analysisConfig.analysis_depth === profile ? "active" : ""}
                  key={profile}
                  onClick={() => applyPreset(profile)}
                >
                  <strong>{analysisPresetDetails[profile].label}</strong>
                  <span>{analysisPresetDetails[profile].tag}</span>
                </button>
              ))}
            </div>
            <div className="depth-advice">
              {analysisConfig.analysis_depth === "custom" ? (
                <>
                  <strong>自定义参数</strong>
                  <p>你已手动调整预设。阈值越低、切块越细、重叠和候选上限越高，检查越深入，但耗时和待复核候选也会增加。</p>
                </>
              ) : (
                <>
                  <strong>{analysisPresetDetails[analysisConfig.analysis_depth as AnalysisPreset].description}</strong>
                  <p>{analysisPresetDetails[analysisConfig.analysis_depth as AnalysisPreset].recommendation}</p>
                </>
              )}
            </div>
            <div className="settings-grid">
              <ParameterInput label="文本确认阈值" description="两段文字达到多像才算匹配。越低越容易发现改写内容。" value={analysisConfig.text_threshold} step={0.01} min={0.4} max={0.95} onChange={(value) => updateAnalysisConfig({ text_threshold: value })} />
              <ParameterInput label="图片确认阈值" description="两张图片达到多像才算匹配。越低越容易识别压缩或缩放图片。" value={analysisConfig.image_threshold} step={0.01} min={0.6} max={0.95} onChange={(value) => updateAnalysisConfig({ image_threshold: value })} />
              <ParameterInput label="成组阈值" description="文件综合分达到多少才进入雷同组。越低会显示更多弱关联。" value={finalThreshold} step={0.01} min={0.3} max={0.95} onChange={(value) => updateAnalysisConfig({ final_threshold: value })} />
              <ParameterInput label="最小块长度" description="短于多少字的文本块会被忽略。越小越容易发现零散复制。" value={analysisConfig.min_chunk_chars} step={10} min={30} max={500} onChange={(value) => updateAnalysisConfig({ min_chunk_chars: value })} />
              <ParameterInput label="目标块长度" description="每段文字大约切成多少字来比较。越小检查越细。" value={analysisConfig.target_chunk_chars} step={20} min={200} max={1000} onChange={(value) => updateAnalysisConfig({ target_chunk_chars: value })} />
              <ParameterInput label="块重叠字符" description="相邻文本块重复保留的字数，避免相似内容刚好被切断。" value={analysisConfig.chunk_overlap_chars} step={10} min={20} max={300} onChange={(value) => updateAnalysisConfig({ chunk_overlap_chars: value })} />
              <ParameterInput label="shingle 粒度" description="连续几个字组成一个文本指纹。越小越敏感，也更容易误报。" value={analysisConfig.shingle_size} step={1} min={3} max={12} onChange={(value) => updateAnalysisConfig({ shingle_size: value })} />
              <ParameterInput label="共享 shingle 下限" description="至少共享多少个文本指纹，才进入下一轮检查。越小召回越多。" value={analysisConfig.min_shared_shingles} step={1} min={1} max={20} onChange={(value) => updateAnalysisConfig({ min_shared_shingles: value })} />
              <ParameterInput label="SimHash 容差" description="允许文本摘要指纹相差多少位。越大越能容忍改写。" value={analysisConfig.simhash_hamming_threshold} step={1} min={0} max={16} onChange={(value) => updateAnalysisConfig({ simhash_hamming_threshold: value })} />
              <ParameterInput label="候选召回阈值" description="初筛分达到多少才深入比较。越低漏检更少，但耗时更长。" value={analysisConfig.candidate_score_threshold} step={0.01} min={0.05} max={0.8} onChange={(value) => updateAnalysisConfig({ candidate_score_threshold: value })} />
              <ParameterInput label="每文件候选上限" description="每份文件最多与多少份其他文件深入比较。越高检查越全面。" value={analysisConfig.candidate_top_k_per_file} step={1} min={4} max={100} onChange={(value) => updateAnalysisConfig({ candidate_top_k_per_file: value })} />
              <ParameterInput label="每对证据上限" description="每对文件最多保留多少条匹配证据。越高报告越完整。" value={analysisConfig.max_matches_per_pair} step={5} min={5} max={200} onChange={(value) => updateAnalysisConfig({ max_matches_per_pair: value })} />
            </div>
            <div className="settings-foot">
              <span>当前模式：{analysisConfig.analysis_depth === "custom" ? "自定义参数" : analysisPresetDetails[analysisConfig.analysis_depth as AnalysisPreset].label}</span>
              <button className="button primary" onClick={() => setSettingsOpen(false)}>
                <Check size={15} />
                应用参数
              </button>
            </div>
          </section>
        </div>
      )}

      <section className="workspace">
        <aside className="left-rail">
          <RailHeading title="环境自检" action="分析前检查" />
          <section className="rail-card checks">
            <CheckRow title="lopdf 文本引擎" copy="逐页解析，中文路径可用" status="通过" />
            <CheckRow title="Tauri 权限" copy="dialog / 本地文件可用" status="通过" />
            <CheckRow title="SQLite 缓存" copy="断点恢复尚未接入" status="未启用" muted />
          </section>

          <RailHeading title="分析流水线" action={currentStep ? `当前第 ${currentStep} 步` : "等待开始"} />
          <section className="rail-card pipeline">
            {pipeline.map(([title, copy], index) => {
              const step = index + 1;
              const complete = currentStep > step || progress?.stage === "Done";
              const active = currentStep === step && progress?.stage !== "Done";
              return (
                <div className="pipeline-row" key={title}>
                  <b className={active ? "step active" : complete ? "step complete" : "step"}>{step}</b>
                  <div>
                    <strong>{title}</strong>
                    <span>{copy}</span>
                  </div>
                  <em className={active ? "run" : complete ? "ok" : ""}>
                    {active ? "执行中" : complete ? "完成" : "待执行"}
                  </em>
                </div>
              );
            })}
          </section>

          <RailHeading title="导入文件" action={`${paths.length} 个 PDF`} />
          <section className="file-stack">
            {paths.length === 0 && <Empty copy="导入 PDF 后显示文件状态。" />}
            {paths.map((path) => {
              const file = files.find((item) => item.path === path);
              return (
                <article className="file-card" key={path}>
                  <div className="file-title">
                    <strong>{file?.file_name ?? basename(path)}</strong>
                    <span className={`mini-status ${file?.status === "failed" ? "red" : ""}`}>
                      {file ? statusLabel(file.status) : isRunning ? "扫描中" : "待分析"}
                    </span>
                  </div>
                  {file ? (
                    <div className="file-metrics">
                      <Metric value={number(file.page_count)} label="页" />
                      <Metric value={chars(file.total_text_chars)} label="字" />
                      <Metric value={number(file.indexed_image_count)} label="有效图" />
                    </div>
                  ) : (
                    <p className="path">{path}</p>
                  )}
                </article>
              );
            })}
          </section>
        </aside>

        <section className="center-stage">
          <div className="overview-grid">
            <section className="panel progress-card">
              <PanelHeading title="详细执行进度" action={progress ? stageLabels[progress.stage] : "Idle"} />
              <div className="progress-hero">
                <div>
                  <h2>{progress?.message ?? "导入至少 2 个 PDF 开始本地分析"}</h2>
                  <p>
                    {progress?.current_file
                      ? `${progress.current_file}${progress.current_page ? ` · 第 ${progress.current_page} 页` : ""}`
                      : "文本索引和证据片段只保留在本机。"}
                  </p>
                </div>
                <b>{percent(progress)}%</b>
              </div>
              <ProgressBar value={percent(progress)} />
              <div className="progress-stats">
                <Metric value={number(progress?.processed_pages ?? 0)} label="已处理页" />
                <Metric value={number(progress?.candidate_pairs ?? 0)} label="候选对" />
                <Metric value={number(progress?.confirmed_pairs ?? 0)} label="确认关系" />
                <Metric value={number(progress?.similarity_groups ?? 0)} label="雷同组" />
                <Metric
                  value={progress?.estimated_remaining_seconds ? `${progress.estimated_remaining_seconds}s` : "--"}
                  label="预计剩余"
                />
              </div>
            </section>

            <section className="panel index-card">
              <PanelHeading title="索引召回概览" action="全局索引" />
              <IndexRow color="blue" label="文本指纹" value={number(result ? totalChunks : (progress?.indexed_chunks ?? 0))} width={(result || progress?.indexed_chunks) ? 84 : 0} />
              <IndexRow color="green" label="图片指纹" value={number(result ? indexedImages : (progress?.indexed_images ?? 0))} width={(result || progress?.indexed_images) ? 56 : 0} />
              <IndexRow color="amber" label="处理页面" value={number(progress?.total_pages ?? 0)} width={progress ? 72 : 0} />
              <IndexRow color="purple" label="候选召回" value={`${progress?.candidate_pairs ?? 0} 对`} width={progress ? 66 : 0} />
              <div className="index-note">已扫描 {number(totalImages)} 个图片对象；小图标会过滤，公共 Logo 会降噪。</div>
            </section>
          </div>

          <section className="panel result-panel">
            <div className="result-columns">
              <section className="group-column">
                <PanelHeading title="雷同组" action="主结果入口" />
                <div className="group-list">
                  {groups.length === 0 && <Empty copy="分析完成后在这里展示雷同组。" />}
                  {groups.map((group, index) => (
                    <button
                      className={`group-card ${selectedGroup?.group_id === group.group_id ? "selected" : ""}`}
                      key={group.group_id}
                      onClick={() => setSelectedGroupId(group.group_id)}
                    >
                      <div className="group-card-title">
                        <strong>雷同组 {index + 1}</strong>
                        <span className={group.group_score >= 0.85 ? "score-chip red" : "score-chip amber"}>
                          {group.group_score.toFixed(2)}
                        </span>
                      </div>
                      <p>{group.files.map((file) => compactName(file, 10)).join("、")}</p>
                      <div className="group-metrics">
                        <Metric value={number(group.files.length)} label="文件" />
                        <Metric
                          value={number(
                            group.pair_relations.reduce(
                              (sum, relation) =>
                                sum +
                                (sortedPairs.find(
                                  (pair) =>
                                    pair.left_file_id === relation.left_file_id &&
                                    pair.right_file_id === relation.right_file_id,
                                )?.exact_page_match_count ?? 0),
                              0,
                            ),
                          )}
                          label="精确页"
                        />
                        <Metric value={group.graph_density.toFixed(2)} label="密度" />
                        <Metric value={levelLabel(group.level)} label="等级" />
                      </div>
                    </button>
                  ))}
                </div>
              </section>

              <section className="relation-column">
                <PanelHeading title="组内相似图与关系明细" action="connected components" />
                <GroupGraph group={selectedGroup} />
                <div className="relation-table">
                  <div className="relation-head">
                    <span>组内关系</span>
                    <span>综合</span>
                    <span>文本</span>
                    <span>图片</span>
                    <span>页面</span>
                    <span>状态</span>
                  </div>
                  {relationPairs.map((pair) => (
                      <button
                        className={`relation-row ${selectedPair?.pair_id === pair.pair_id ? "selected" : ""}`}
                        key={pair.pair_id}
                        onClick={() => setSelectedPairId(pair.pair_id)}
                      >
                        <span>{compactName(pair.left_file, 7)} ↔ {compactName(pair.right_file, 7)}</span>
                        <b>{score(pair.final_score)}</b>
                        <span>{score(pair.text_score)}</span>
                        <span>{score(pair.image_score)}</span>
                        <span>--</span>
                        <em>已确认</em>
                      </button>
                    ))}
                  {relationPairs.length === 0 && <Empty copy="当前雷同组没有可展示的确认关系。" />}
                </div>
              </section>
            </div>
          </section>
        </section>

        <aside className="right-rail">
          <section className="right-title">
            <div>
              <h2>{selectedGroup ? `雷同组 · ${selectedGroup.files.length} 个文件` : "雷同组详情"}</h2>
              <p>{selectedGroup ? selectedGroup.files.map((file) => compactName(file, 8)).join("、") : "选择雷同组后查看证据。"}</p>
            </div>
            {selectedGroup && <span className="score-chip red">{levelLabel(selectedGroup.level)}</span>}
          </section>

          <section className="summary-grid">
            <Metric value={number(selectedGroup?.files.length ?? 0)} label="组内文件" />
            <Metric value={number(totalEvidence)} label="代表证据" />
            <Metric value={selectedGroup?.graph_density.toFixed(2) ?? "--"} label="关系密度" />
          </section>

          <RightPanel title="共同证据摘要" action="Top 证据">
            <SummaryRow label="共同文本" value={selectedPair ? `${chars(selectedPair.matched_text_chars)} 字覆盖` : "--"} />
            <SummaryRow label="精确页面" value={selectedPair ? `${number(selectedPair.exact_page_match_count)} 页` : "--"} />
            <SummaryRow
              label="近似片段"
              value={selectedPair ? `${number(selectedPair.approximate_text_match_count)} 处` : "--"}
            />
            <SummaryRow label="图片证据" value={selectedPair ? `${number(selectedPair.matched_images.length)} 张` : "--"} />
          </RightPanel>

          <RightPanel title="代表雷同文本" action={selectedPair ? "文件对证据" : "等待选择"}>
            {selectedPair?.matched_texts[0] ? (
              <>
                {selectedPair.matched_texts[0].text_readable ? (
                  <blockquote>{selectedPair.matched_texts[0].left_text}</blockquote>
                ) : (
                  <div className="warning-box">
                    <AlertTriangle size={15} />
                    PDF 缺少 ToUnicode 字体映射，无法展示可读中文。当前关系来自 CID 字形指纹比对，仍可用于判断雷同性。
                  </div>
                )}
                <SummaryRow
                  label="出现位置"
                  value={`A P${selectedPair.matched_texts[0].left_page} / B P${selectedPair.matched_texts[0].right_page}`}
                />
                <SummaryRow
                  label="证据类型"
                  value={
                    selectedPair.matched_texts[0].text_readable
                      ? selectedPair.exact_page_match_count
                        ? "精确页面"
                        : "近似文本"
                      : "CID 字形指纹"
                  }
                />
              </>
            ) : (
              <Empty copy="选择关系后查看代表文本。" />
            )}
          </RightPanel>

          <RightPanel title="代表雷同图片" action={selectedPair?.matched_images[0] ? "图片证据" : "等待选择"}>
            {selectedPair?.matched_images[0] ? (
              <>
                <SummaryRow
                  label="出现位置"
                  value={`A P${selectedPair.matched_images[0].left_page} / B P${selectedPair.matched_images[0].right_page}`}
                />
                <SummaryRow
                  label="证据类型"
                  value={selectedPair.matched_images[0].exact ? "完全重复" : "近似重复"}
                />
                <SummaryRow
                  label="pHash 距离"
                  value={number(selectedPair.matched_images[0].hamming_distance)}
                />
                <SummaryRow
                  label="有效尺寸"
                  value={`${selectedPair.matched_images[0].width} × ${selectedPair.matched_images[0].height}`}
                />
              </>
            ) : (
              <Empty copy="当前关系没有图片证据。" />
            )}
          </RightPanel>

          <RightPanel title="能力提示" action="内嵌图片版">
            <div className="warning-box">
              <AlertTriangle size={15} />
              内嵌图片 SHA-256 与 pHash 已启用。扫描件页面渲染、OCR 和 SQLite 缓存尚未接入。
            </div>
          </RightPanel>

          <RightPanel title="导出配置" action="敏感确认">
            <div className="export-grid">
              <label><input type="checkbox" checked={exportWord} onChange={(event) => setExportWord(event.target.checked)} /> Word</label>
              <label><input type="checkbox" checked={exportJson} onChange={(event) => setExportJson(event.target.checked)} /> JSON</label>
              <label><input type="checkbox" checked={includeTextEvidence} onChange={(event) => setIncludeTextEvidence(event.target.checked)} /> 正文片段</label>
              <label><input type="checkbox" disabled /> 图片缩略图</label>
            </div>
            <button className="export-directory" onClick={exportReport} disabled={!result}>
              <FolderOpen size={14} />
              选择目录并导出
            </button>
          </RightPanel>
        </aside>
      </section>
    </main>
  );
}

function RailHeading({ title, action }: { title: string; action: string }) {
  return (
    <div className="rail-heading">
      <strong>{title}</strong>
      <span>{action}</span>
    </div>
  );
}

function PanelHeading({ title, action }: { title: string; action: string }) {
  return (
    <div className="panel-heading">
      <strong>{title}</strong>
      <span>{action}</span>
    </div>
  );
}

function CheckRow({ title, copy, status, muted = false }: { title: string; copy: string; status: string; muted?: boolean }) {
  return (
    <div className="check-row">
      <div>
        <strong>{title}</strong>
        <span>{copy}</span>
      </div>
      <em className={muted ? "muted" : ""}>{status}</em>
    </div>
  );
}

function Metric({ value, label }: { value: string; label: string }) {
  return (
    <div className="metric">
      <b>{value}</b>
      <span>{label}</span>
    </div>
  );
}

function ProgressBar({ value }: { value: number }) {
  return (
    <div className="progress-bar">
      <div style={{ width: `${value}%` }} />
    </div>
  );
}

function IndexRow({ color, label, value, width }: { color: string; label: string; value: string; width: number }) {
  return (
    <div className="index-row">
      <b>{label}</b>
      <div><i className={color} style={{ width: `${width}%` }} /></div>
      <span>{value}</span>
    </div>
  );
}

function SummaryRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="summary-row">
      <span>{label}</span>
      <b>{value}</b>
    </div>
  );
}

function RightPanel({ title, action, children }: { title: string; action: string; children: React.ReactNode }) {
  return (
    <section className="right-panel">
      <div className="right-panel-head">
        <strong>{title}</strong>
        <span>{action}</span>
      </div>
      <div className="right-panel-body">{children}</div>
    </section>
  );
}

function Empty({ copy }: { copy: string }) {
  return <div className="empty">{copy}</div>;
}

function ParameterInput({
  label,
  description,
  value,
  step,
  min,
  max,
  onChange,
}: {
  label: string;
  description: string;
  value: number;
  step: number;
  min: number;
  max: number;
  onChange: (value: number) => void;
}) {
  return (
    <label className="parameter-input">
      <span>{label}</span>
      <input
        type="number"
        value={value}
        min={min}
        max={max}
        step={step}
        onChange={(event) => onChange(Number(event.target.value))}
      />
      <small>{description}</small>
    </label>
  );
}

function GroupGraph({ group }: { group?: SimilarityGroup }) {
  if (!group) return <div className="graph empty">选择雷同组后显示关系图。</div>;
  return (
    <div className="graph">
      <div className="graph-head">
        <div>
          <strong>成员平铺</strong>
          <span>{group.files.length} 个文件 · {group.pair_relations.length} 条关系</span>
        </div>
        <div className="graph-score">
          <span>组评分</span>
          <b>{group.group_score.toFixed(2)}</b>
        </div>
      </div>
      <div className="member-tile-grid">
        {group.files.map((file, index) => (
          <div className="member-tile" title={file} key={group.file_ids[index]}>{compactName(file, 12)}</div>
        ))}
      </div>
      <div className="relation-tile-grid">
        {group.pair_relations.map((relation) => (
          <div className="relation-tile" key={`${relation.left_file_id}-${relation.right_file_id}`}>
            <span title={relation.left_file}>{compactName(relation.left_file, 8)}</span>
            <b>↔</b>
            <span title={relation.right_file}>{compactName(relation.right_file, 8)}</span>
            <em>{score(relation.final_score)}</em>
          </div>
        ))}
      </div>
    </div>
  );
}
