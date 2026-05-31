use chrono::Utc;
use lopdf::content::Content;
use lopdf::{Document, Object};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::hash_map::DefaultHasher;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::hash::{Hash, Hasher};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

const MIN_CHUNK_CHARS: usize = 80;
const TARGET_CHUNK_CHARS: usize = 500;
const CHUNK_OVERLAP_CHARS: usize = 80;
const SHINGLE_SIZE: usize = 5;
const MIN_SHARED_SHINGLES: u32 = 3;
const MAX_COMMON_FEATURE_FILES: usize = 10;
const MAX_POSTINGS_PER_FEATURE: usize = 180;
const MAX_MATCHES_PER_PAIR: usize = 30;
const MIN_EXACT_PAGE_CHARS: usize = 1;
const CANDIDATE_SCORE_THRESHOLD: f32 = 0.35;
const CANDIDATE_TOP_K_PER_FILE: usize = 20;
const CANDIDATE_MIN_CHUNK_PAIRS: usize = 2;
const CANDIDATE_STRONG_SINGLE_CHUNK_SHINGLES: u32 = 16;
const CID_FINGERPRINT_BASE: u32 = 0xF0000;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalyzeRequest {
    pub paths: Vec<String>,
    pub analysis_depth: String,
    pub text_threshold: f32,
    pub image_threshold: f32,
    pub final_threshold: f32,
    pub min_chunk_chars: usize,
    pub target_chunk_chars: usize,
    pub chunk_overlap_chars: usize,
    pub shingle_size: usize,
    pub min_shared_shingles: u32,
    pub simhash_hamming_threshold: u32,
    pub candidate_score_threshold: f32,
    pub candidate_top_k_per_file: usize,
    pub max_matches_per_pair: usize,
}

impl Default for AnalyzeRequest {
    fn default() -> Self {
        Self {
            paths: Vec::new(),
            analysis_depth: "balanced".to_string(),
            text_threshold: 0.72,
            image_threshold: 0.80,
            final_threshold: 0.50,
            min_chunk_chars: MIN_CHUNK_CHARS,
            target_chunk_chars: TARGET_CHUNK_CHARS,
            chunk_overlap_chars: CHUNK_OVERLAP_CHARS,
            shingle_size: SHINGLE_SIZE,
            min_shared_shingles: MIN_SHARED_SHINGLES,
            simhash_hamming_threshold: 4,
            candidate_score_threshold: CANDIDATE_SCORE_THRESHOLD,
            candidate_top_k_per_file: CANDIDATE_TOP_K_PER_FILE,
            max_matches_per_pair: MAX_MATCHES_PER_PAIR,
        }
    }
}

impl AnalyzeRequest {
    pub fn validate(&self) -> Result<(), String> {
        if self.paths.len() < 2 {
            return Err("请至少选择 2 个 PDF 文件。".to_string());
        }
        if !(0.0..=1.0).contains(&self.text_threshold)
            || !(0.0..=1.0).contains(&self.final_threshold)
        {
            return Err("相似度阈值必须位于 0 到 1 之间。".to_string());
        }
        if self.min_chunk_chars < 30 || self.target_chunk_chars < self.min_chunk_chars {
            return Err("文本块长度设置无效。".to_string());
        }
        if self.chunk_overlap_chars >= self.target_chunk_chars {
            return Err("文本块重叠必须小于目标块长度。".to_string());
        }
        if !(3..=12).contains(&self.shingle_size) {
            return Err("shingle 粒度必须位于 3 到 12 之间。".to_string());
        }
        if self.min_shared_shingles == 0
            || self.candidate_top_k_per_file == 0
            || self.max_matches_per_pair == 0
        {
            return Err("召回和证据数量参数必须大于 0。".to_string());
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisProgress {
    pub task_id: String,
    pub stage: AnalysisStage,
    pub current_file: Option<String>,
    pub current_page: Option<usize>,
    pub processed_files: usize,
    pub total_files: usize,
    pub processed_pages: usize,
    pub total_pages: usize,
    pub cache_hits: usize,
    pub candidate_pairs: usize,
    pub confirmed_pairs: usize,
    pub similarity_groups: usize,
    pub weak_connection_groups: usize,
    pub confirmed_text_matches: usize,
    pub confirmed_image_matches: usize,
    pub elapsed_seconds: u64,
    pub estimated_remaining_seconds: Option<u64>,
    pub message: String,
}

impl AnalysisProgress {
    pub fn new(task_id: String, total_files: usize) -> Self {
        Self {
            task_id,
            stage: AnalysisStage::Init,
            current_file: None,
            current_page: None,
            processed_files: 0,
            total_files,
            processed_pages: 0,
            total_pages: 0,
            cache_hits: 0,
            candidate_pairs: 0,
            confirmed_pairs: 0,
            similarity_groups: 0,
            weak_connection_groups: 0,
            confirmed_text_matches: 0,
            confirmed_image_matches: 0,
            elapsed_seconds: 0,
            estimated_remaining_seconds: None,
            message: "任务已创建。".to_string(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AnalysisStage {
    Init,
    ReadingMeta,
    BuildingTextIndex,
    ComparingText,
    GeneratingReport,
    Done,
    Cancelled,
    Failed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisResult {
    pub task_id: String,
    pub started_at: String,
    pub finished_at: Option<String>,
    pub analysis_settings: AnalyzeRequest,
    pub files: Vec<FileSummary>,
    pub pairs: Vec<SimilarityPair>,
    pub groups: Vec<SimilarityGroup>,
    pub report_path: Option<String>,
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportReportRequest {
    pub task_id: String,
    pub target_dir: String,
    pub export_json: bool,
    pub export_word: bool,
    pub include_text_evidence: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportReportResult {
    pub exported_files: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileSummary {
    pub id: String,
    pub path: String,
    pub file_name: String,
    pub page_count: usize,
    pub total_text_chars: usize,
    pub chunk_count: usize,
    pub image_count: usize,
    pub status: String,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimilarityPair {
    pub pair_id: String,
    pub left_file: String,
    pub right_file: String,
    pub text_score: f32,
    pub image_score: f32,
    pub page_image_score: f32,
    pub final_score: f32,
    pub level: SimilarityLevel,
    pub exact_page_match_count: usize,
    pub approximate_text_match_count: usize,
    pub matched_text_chars: usize,
    pub matched_texts: Vec<MatchedText>,
    pub matched_images: Vec<MatchedImage>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatchedText {
    pub left_page: usize,
    pub right_page: usize,
    pub similarity: f32,
    pub text_readable: bool,
    pub left_text: String,
    pub right_text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatchedImage {
    pub left_page: usize,
    pub right_page: usize,
    pub hamming_distance: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimilarityGroup {
    pub group_id: String,
    pub files: Vec<String>,
    pub group_score: f32,
    pub level: SimilarityLevel,
    pub graph_density: f32,
    pub quality_flags: Vec<GroupQualityFlag>,
    pub pair_relations: Vec<PairRelation>,
    pub representative_evidence: Vec<GroupEvidence>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PairRelation {
    pub left_file: String,
    pub right_file: String,
    pub final_score: f32,
    pub text_score: f32,
    pub image_score: f32,
    pub page_image_score: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroupEvidence {
    pub evidence_type: String,
    pub summary: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GroupQualityFlag {
    WeakConnection,
    NeedsManualReview,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SimilarityLevel {
    Extreme,
    High,
    Medium,
    Low,
}

#[derive(Clone)]
struct PdfDocumentData {
    summary: FileSummary,
    pages: Vec<PageText>,
    chunks: Vec<TextChunk>,
    extraction_warnings: Vec<String>,
}

#[derive(Clone)]
struct PageText {
    page: usize,
    text: String,
    text_hash: u64,
    char_count: usize,
}

#[derive(Clone)]
struct TextChunk {
    page: usize,
    start: usize,
    end: usize,
    text: String,
    text_hash: u64,
    shingles: Vec<u64>,
    shingle_set: HashSet<u64>,
    simhash: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct ChunkRef {
    file_index: usize,
    chunk_index: usize,
}

pub fn run_analysis(
    request: AnalyzeRequest,
    task_id: String,
    on_progress: impl Fn(AnalysisProgress),
    should_cancel: impl Fn() -> bool,
) -> AnalysisResult {
    let started_at = Utc::now().to_rfc3339();
    let timer = Instant::now();
    let mut progress = AnalysisProgress::new(task_id.clone(), request.paths.len());
    let mut docs = Vec::new();
    let mut warnings =
        vec!["当前版本仅启用文本检测；图片 pHash、扫描件页面渲染和 OCR 尚未启用。".to_string()];

    progress.stage = AnalysisStage::ReadingMeta;
    progress.message = "正在逐页读取 PDF 文本。".to_string();
    update_progress(&mut progress, timer, &on_progress);

    for (file_index, path) in request.paths.iter().enumerate() {
        if should_cancel() {
            progress.stage = AnalysisStage::Cancelled;
            progress.message = "任务已取消。".to_string();
            update_progress(&mut progress, timer, &on_progress);
            break;
        }

        progress.current_file = Some(file_name(path));
        progress.current_page = None;
        update_progress(&mut progress, timer, &on_progress);

        match extract_pdf(
            file_index,
            path,
            &request,
            &mut progress,
            timer,
            &on_progress,
            &should_cancel,
        ) {
            Ok(doc) => {
                progress.processed_files += 1;
                progress.processed_pages += doc.summary.page_count;
                warnings.extend(
                    doc.extraction_warnings
                        .iter()
                        .map(|warning| format!("{}: {warning}", doc.summary.file_name)),
                );
                docs.push(doc);
            }
            Err(error) => {
                progress.processed_files += 1;
                warnings.push(format!("{}: {}", file_name(path), error));
                docs.push(PdfDocumentData {
                    summary: FileSummary {
                        id: format!("file-{file_index}"),
                        path: path.to_string(),
                        file_name: file_name(path),
                        page_count: 0,
                        total_text_chars: 0,
                        chunk_count: 0,
                        image_count: 0,
                        status: "failed".to_string(),
                        error: Some(error),
                    },
                    pages: Vec::new(),
                    chunks: Vec::new(),
                    extraction_warnings: Vec::new(),
                });
            }
        }
        update_progress(&mut progress, timer, &on_progress);
    }

    progress.stage = AnalysisStage::BuildingTextIndex;
    progress.current_file = None;
    progress.current_page = None;
    progress.message = "正在建立 shingle 倒排索引。".to_string();
    update_progress(&mut progress, timer, &on_progress);

    let (candidate_scores, chunk_pair_features) = build_text_index(&docs, &request);
    progress.candidate_pairs = candidate_scores.len();
    progress.message = format!("已召回 {} 个候选文件对。", progress.candidate_pairs);
    update_progress(&mut progress, timer, &on_progress);

    progress.stage = AnalysisStage::ComparingText;
    progress.message = "正在对候选文件对进行文本精算。".to_string();
    update_progress(&mut progress, timer, &on_progress);

    let mut pairs = compare_candidates(&docs, &request, &candidate_scores, &chunk_pair_features);
    pairs.sort_by(|left, right| {
        right
            .final_score
            .partial_cmp(&left.final_score)
            .unwrap_or(Ordering::Equal)
    });
    progress.confirmed_pairs = pairs.len();
    progress.confirmed_text_matches = pairs.iter().map(|pair| pair.matched_texts.len()).sum();
    update_progress(&mut progress, timer, &on_progress);

    progress.stage = AnalysisStage::GeneratingReport;
    progress.message = "正在聚类雷同组并写入 JSON 报告。".to_string();
    update_progress(&mut progress, timer, &on_progress);

    let groups = build_groups(&pairs, &request);
    progress.similarity_groups = groups.len();
    progress.weak_connection_groups = groups
        .iter()
        .filter(|group| {
            group
                .quality_flags
                .iter()
                .any(|flag| matches!(flag, GroupQualityFlag::WeakConnection))
        })
        .count();

    let mut result = AnalysisResult {
        task_id: task_id.clone(),
        started_at,
        finished_at: Some(Utc::now().to_rfc3339()),
        analysis_settings: request.clone(),
        files: docs.iter().map(|doc| doc.summary.clone()).collect(),
        pairs,
        groups,
        report_path: None,
        warnings,
    };

    match write_report(&result) {
        Ok(path) => result.report_path = Some(path.to_string_lossy().to_string()),
        Err(error) => result.warnings.push(format!("报告写入失败: {error}")),
    }

    progress.stage = if should_cancel() {
        AnalysisStage::Cancelled
    } else {
        AnalysisStage::Done
    };
    progress.message = if should_cancel() {
        "任务已取消，当前结果已生成。".to_string()
    } else {
        "分析完成。".to_string()
    };
    update_progress(&mut progress, timer, &on_progress);

    result
}

fn extract_pdf(
    file_index: usize,
    path: &str,
    request: &AnalyzeRequest,
    progress: &mut AnalysisProgress,
    timer: Instant,
    on_progress: &impl Fn(AnalysisProgress),
    should_cancel: &impl Fn() -> bool,
) -> Result<PdfDocumentData, String> {
    let document = Document::load(path).map_err(|error| format!("无法打开 PDF: {error}"))?;
    let pages = document.get_pages();
    let page_count = pages.len();
    progress.total_pages += page_count;
    progress.message = format!("正在读取 {}，共 {} 页。", file_name(path), page_count);
    update_progress(progress, timer, on_progress);

    let image_count = count_images(&document);
    let mut raw_pages = Vec::with_capacity(page_count);
    let mut used_cid_fallback = false;
    let mut extraction_errors = Vec::new();

    for (page_offset, page_number) in pages.keys().enumerate() {
        if should_cancel() {
            break;
        }

        let page = page_offset + 1;
        progress.current_page = Some(page);
        if page == 1 || page % 10 == 0 {
            progress.message = format!("正在读取 {} 第 {} 页。", file_name(path), page);
            update_progress(progress, timer, on_progress);
        }

        let raw_text = match document.extract_text(&[*page_number]) {
            Ok(text) => text,
            Err(error) => match extract_identity_h_cid_text(&document, *page_number) {
                Ok(text) if !text.is_empty() => {
                    used_cid_fallback = true;
                    text
                }
                Ok(_) => {
                    if extraction_errors.len() < 3 {
                        extraction_errors.push(format!("第 {page} 页文本抽取失败: {error}"));
                    }
                    String::new()
                }
                Err(fallback_error) => {
                    if extraction_errors.len() < 3 {
                        extraction_errors.push(format!(
                            "第 {page} 页文本抽取失败: {error}; CID 回退失败: {fallback_error}"
                        ));
                    }
                    String::new()
                }
            },
        };
        raw_pages.push((page, raw_text));
    }

    let pages = clean_page_texts(raw_pages);
    let total_text_chars = pages.iter().map(|page| page.char_count).sum();
    let chunks = pages
        .iter()
        .flat_map(|page| chunk_text(&page.text, page.page, request))
        .collect::<Vec<_>>();

    let status = if total_text_chars == 0 {
        "text-empty"
    } else if used_cid_fallback {
        "cid-fallback"
    } else {
        "ready"
    };

    let mut extraction_warnings = extraction_errors;
    if used_cid_fallback {
        extraction_warnings.push(
            "PDF 缺少中文 CID 字体的 ToUnicode 映射，已使用 CID 字形序列生成指纹；字数为字形数，证据预览无法还原可读中文。"
                .to_string(),
        );
    }

    Ok(PdfDocumentData {
        summary: FileSummary {
            id: format!("file-{file_index}"),
            path: path.to_string(),
            file_name: file_name(path),
            page_count,
            total_text_chars,
            chunk_count: chunks.len(),
            image_count,
            status: status.to_string(),
            error: None,
        },
        pages,
        chunks,
        extraction_warnings,
    })
}

fn extract_identity_h_cid_text(document: &Document, page_number: u32) -> Result<String, String> {
    let pages = document.get_pages();
    let page_id = pages
        .get(&page_number)
        .copied()
        .ok_or_else(|| format!("页码不存在: {page_number}"))?;
    let cid_fonts = document
        .get_page_fonts(page_id)
        .map_err(|error| error.to_string())?
        .into_iter()
        .filter_map(|(name, font)| {
            let is_identity_h = matches!(
                font.get(b"Encoding"),
                Ok(Object::Name(encoding)) if encoding == b"Identity-H"
            );
            let has_to_unicode = font.get(b"ToUnicode").is_ok();
            (is_identity_h && !has_to_unicode).then_some(name)
        })
        .collect::<HashSet<_>>();

    if cid_fonts.is_empty() {
        return Err("没有可回退的 Identity-H CID 字体".to_string());
    }

    let content_data = document
        .get_page_content(page_id)
        .map_err(|error| error.to_string())?;
    let content = Content::decode(&content_data).map_err(|error| error.to_string())?;
    let mut current_font_is_cid = false;
    let mut text = String::new();

    for operation in content.operations {
        match operation.operator.as_str() {
            "Tf" => {
                current_font_is_cid = operation
                    .operands
                    .first()
                    .and_then(|operand| operand.as_name().ok())
                    .map(|font| cid_fonts.contains(font))
                    .unwrap_or(false);
            }
            "Tj" | "TJ" if current_font_is_cid => {
                collect_identity_h_cids(&operation.operands, &mut text);
            }
            "ET" if current_font_is_cid && !text.ends_with('\n') => text.push('\n'),
            _ => {}
        }
    }

    Ok(text)
}

fn collect_identity_h_cids(operands: &[Object], text: &mut String) {
    for operand in operands {
        match operand {
            Object::String(bytes, _) => text.push_str(&identity_h_cids_to_fingerprint(bytes)),
            Object::Array(items) => collect_identity_h_cids(items, text),
            _ => {}
        }
    }
}

fn identity_h_cids_to_fingerprint(bytes: &[u8]) -> String {
    bytes
        .chunks(2)
        .filter_map(|chunk| {
            let cid = if chunk.len() == 2 {
                u16::from_be_bytes([chunk[0], chunk[1]]) as u32
            } else {
                chunk[0] as u32
            };
            char::from_u32(CID_FINGERPRINT_BASE + cid)
        })
        .collect()
}

fn build_text_index(
    docs: &[PdfDocumentData],
    request: &AnalyzeRequest,
) -> (
    HashMap<(usize, usize), f32>,
    HashMap<(usize, usize, usize, usize), u32>,
) {
    let mut postings: HashMap<u64, Vec<ChunkRef>> = HashMap::new();

    for (file_index, doc) in docs.iter().enumerate() {
        for (chunk_index, chunk) in doc.chunks.iter().enumerate() {
            for hash in &chunk.shingles {
                postings.entry(*hash).or_default().push(ChunkRef {
                    file_index,
                    chunk_index,
                });
            }
        }
    }

    let total_files = docs.len().max(1) as f32;
    let max_common_feature_files = if docs.len() >= 4 {
        ((docs.len() + 1) / 2).min(MAX_COMMON_FEATURE_FILES)
    } else {
        docs.len().min(MAX_COMMON_FEATURE_FILES)
    };
    let mut candidate_scores: HashMap<(usize, usize), f32> = HashMap::new();
    let mut chunk_pair_features: HashMap<(usize, usize, usize, usize), u32> = HashMap::new();

    for refs in postings.values() {
        if refs.len() < 2 || refs.len() > MAX_POSTINGS_PER_FEATURE {
            continue;
        }

        let doc_frequency = refs
            .iter()
            .map(|item| item.file_index)
            .collect::<HashSet<_>>()
            .len();
        if doc_frequency < 2 || doc_frequency > max_common_feature_files {
            continue;
        }

        let idf = (total_files / (1.0 + doc_frequency as f32)).ln().max(0.1);
        for left_index in 0..refs.len() {
            for right_index in (left_index + 1)..refs.len() {
                let left = refs[left_index];
                let right = refs[right_index];
                if left.file_index == right.file_index {
                    continue;
                }

                let (file_left, chunk_left, file_right, chunk_right) =
                    if left.file_index < right.file_index {
                        (
                            left.file_index,
                            left.chunk_index,
                            right.file_index,
                            right.chunk_index,
                        )
                    } else {
                        (
                            right.file_index,
                            right.chunk_index,
                            left.file_index,
                            left.chunk_index,
                        )
                    };

                *candidate_scores.entry((file_left, file_right)).or_default() += idf;
                *chunk_pair_features
                    .entry((file_left, chunk_left, file_right, chunk_right))
                    .or_default() += 1;
            }
        }
    }

    let chunk_counts = docs.iter().map(|doc| doc.chunks.len()).collect::<Vec<_>>();
    let candidate_scores = select_candidate_pairs(
        &candidate_scores,
        &chunk_pair_features,
        &chunk_counts,
        request,
    );
    let chunk_pair_features = chunk_pair_features
        .into_iter()
        .filter(|((left_file, _, right_file, _), _)| {
            candidate_scores.contains_key(&(*left_file, *right_file))
        })
        .collect();

    (candidate_scores, chunk_pair_features)
}

fn select_candidate_pairs(
    raw_scores: &HashMap<(usize, usize), f32>,
    chunk_pair_features: &HashMap<(usize, usize, usize, usize), u32>,
    chunk_counts: &[usize],
    request: &AnalyzeRequest,
) -> HashMap<(usize, usize), f32> {
    #[derive(Default)]
    struct EvidenceSummary {
        chunk_pairs: usize,
        strongest_chunk: u32,
        left_chunks: HashSet<usize>,
        right_chunks: HashSet<usize>,
    }

    let mut evidence: HashMap<(usize, usize), EvidenceSummary> = HashMap::new();
    for ((left_file, left_chunk, right_file, right_chunk), shared) in chunk_pair_features {
        if *shared < request.min_shared_shingles {
            continue;
        }

        let summary = evidence.entry((*left_file, *right_file)).or_default();
        summary.chunk_pairs += 1;
        summary.strongest_chunk = summary.strongest_chunk.max(*shared);
        summary.left_chunks.insert(*left_chunk);
        summary.right_chunks.insert(*right_chunk);
    }

    let max_raw_score = raw_scores.values().copied().fold(0.0f32, f32::max).max(1.0);
    let mut candidates = evidence
        .into_iter()
        .filter_map(|(pair, summary)| {
            let has_enough_evidence = summary.chunk_pairs >= CANDIDATE_MIN_CHUNK_PAIRS
                || summary.strongest_chunk >= CANDIDATE_STRONG_SINGLE_CHUNK_SHINGLES;
            if !has_enough_evidence {
                return None;
            }

            let raw_score = raw_scores.get(&pair).copied().unwrap_or_default();
            let raw_normalized = (raw_score / max_raw_score).min(1.0);
            let left_coverage = summary.left_chunks.len() as f32
                / chunk_counts.get(pair.0).copied().unwrap_or(1).max(1) as f32;
            let right_coverage = summary.right_chunks.len() as f32
                / chunk_counts.get(pair.1).copied().unwrap_or(1).max(1) as f32;
            let document_coverage = left_coverage.min(right_coverage).min(1.0);
            let evidence_volume = (summary.chunk_pairs as f32 / 12.0).min(1.0);
            let candidate_score =
                raw_normalized * 0.15 + document_coverage * 0.75 + evidence_volume * 0.10;
            let has_strong_local_match =
                summary.strongest_chunk >= CANDIDATE_STRONG_SINGLE_CHUNK_SHINGLES * 4;

            (candidate_score >= request.candidate_score_threshold || has_strong_local_match)
                .then_some((pair, candidate_score))
        })
        .collect::<HashMap<_, _>>();

    let mut ranked_per_file: HashMap<usize, Vec<((usize, usize), f32)>> = HashMap::new();
    for (pair @ (left_file, right_file), score) in &candidates {
        ranked_per_file
            .entry(*left_file)
            .or_default()
            .push((*pair, *score));
        ranked_per_file
            .entry(*right_file)
            .or_default()
            .push((*pair, *score));
    }

    let mut selected = HashSet::new();
    for ranked in ranked_per_file.values_mut() {
        ranked.sort_by(|left, right| right.1.partial_cmp(&left.1).unwrap_or(Ordering::Equal));
        selected.extend(
            ranked
                .iter()
                .take(request.candidate_top_k_per_file)
                .map(|(pair, _)| *pair),
        );
    }

    candidates.retain(|pair, _| selected.contains(pair));
    candidates
}

fn compare_candidates(
    docs: &[PdfDocumentData],
    request: &AnalyzeRequest,
    candidate_scores: &HashMap<(usize, usize), f32>,
    chunk_pair_features: &HashMap<(usize, usize, usize, usize), u32>,
) -> Vec<SimilarityPair> {
    let mut pairs = Vec::new();
    let mut by_file_pair: HashMap<(usize, usize), Vec<(usize, usize, u32)>> = HashMap::new();

    for ((left_file, left_chunk, right_file, right_chunk), shared) in chunk_pair_features {
        if *shared >= request.min_shared_shingles {
            by_file_pair
                .entry((*left_file, *right_file))
                .or_default()
                .push((*left_chunk, *right_chunk, *shared));
        }
    }

    for ((left_file, right_file), score) in candidate_scores {
        let Some(candidates) = by_file_pair.get(&(*left_file, *right_file)) else {
            continue;
        };

        let left_doc = &docs[*left_file];
        let right_doc = &docs[*right_file];
        if left_doc.chunks.is_empty() || right_doc.chunks.is_empty() {
            continue;
        }

        let mut ranked = candidates.clone();
        ranked.sort_by(|left, right| right.2.cmp(&left.2));

        let exact_pages = match_exact_pages(left_doc, right_doc);
        let exact_left_pages = exact_pages
            .iter()
            .map(|(left_page, _)| *left_page)
            .collect::<HashSet<_>>();
        let exact_right_pages = exact_pages
            .iter()
            .map(|(_, right_page)| *right_page)
            .collect::<HashSet<_>>();
        let mut left_coverage: HashMap<usize, Vec<(usize, usize)>> = HashMap::new();
        let mut right_coverage: HashMap<usize, Vec<(usize, usize)>> = HashMap::new();
        let mut used_left = HashSet::new();
        let mut used_right = HashSet::new();
        let mut matched_texts = Vec::new();
        let mut approximate_text_match_count = 0usize;

        for (left_page, right_page) in &exact_pages {
            let left_page_text = &left_doc.pages[*left_page];
            let right_page_text = &right_doc.pages[*right_page];
            add_coverage(
                &mut left_coverage,
                left_page_text.page,
                0,
                left_page_text.char_count,
            );
            add_coverage(
                &mut right_coverage,
                right_page_text.page,
                0,
                right_page_text.char_count,
            );
            if matched_texts.len() < request.max_matches_per_pair {
                matched_texts.push(MatchedText {
                    left_page: left_page_text.page,
                    right_page: right_page_text.page,
                    similarity: 1.0,
                    text_readable: text_is_readable(&left_page_text.text)
                        && text_is_readable(&right_page_text.text),
                    left_text: preview(&left_page_text.text),
                    right_text: preview(&right_page_text.text),
                });
            }
        }

        for (left_chunk_index, right_chunk_index, shared) in ranked.into_iter().take(2_000) {
            if used_left.contains(&left_chunk_index) || used_right.contains(&right_chunk_index) {
                continue;
            }

            let left_chunk = &left_doc.chunks[left_chunk_index];
            let right_chunk = &right_doc.chunks[right_chunk_index];
            if exact_left_pages.contains(&(left_chunk.page - 1))
                || exact_right_pages.contains(&(right_chunk.page - 1))
            {
                continue;
            }
            let jaccard = jaccard_similarity(&left_chunk.shingle_set, &right_chunk.shingle_set);
            let simhash_close = hamming_distance(left_chunk.simhash, right_chunk.simhash)
                <= request.simhash_hamming_threshold;
            let shared_ratio = shared as f32
                / left_chunk
                    .shingle_set
                    .len()
                    .min(right_chunk.shingle_set.len())
                    .max(1) as f32;
            let exact_chunk = left_chunk.text_hash == right_chunk.text_hash;
            let is_match = exact_chunk
                || jaccard >= request.text_threshold
                || (simhash_close && shared_ratio >= 0.55);

            if !is_match {
                continue;
            }

            used_left.insert(left_chunk_index);
            used_right.insert(right_chunk_index);
            approximate_text_match_count += 1;
            add_coverage(
                &mut left_coverage,
                left_chunk.page,
                left_chunk.start,
                left_chunk.end,
            );
            add_coverage(
                &mut right_coverage,
                right_chunk.page,
                right_chunk.start,
                right_chunk.end,
            );

            if matched_texts.len() < request.max_matches_per_pair {
                matched_texts.push(MatchedText {
                    left_page: left_chunk.page,
                    right_page: right_chunk.page,
                    similarity: jaccard.max(shared_ratio).min(1.0),
                    text_readable: text_is_readable(&left_chunk.text)
                        && text_is_readable(&right_chunk.text),
                    left_text: preview(&left_chunk.text),
                    right_text: preview(&right_chunk.text),
                });
            }
        }

        if matched_texts.is_empty() {
            continue;
        }

        let min_chars = left_doc
            .summary
            .total_text_chars
            .min(right_doc.summary.total_text_chars)
            .max(1);
        let matched_chars = covered_chars(&left_coverage).min(covered_chars(&right_coverage));
        let text_score = (matched_chars as f32 / min_chars as f32).min(1.0);
        let image_score = 0.0;
        let page_image_score = 0.0;
        let final_score = text_score;

        if final_score < 0.03 && *score < 1.0 {
            continue;
        }

        pairs.push(SimilarityPair {
            pair_id: format!("pair-{}-{}", left_doc.summary.id, right_doc.summary.id),
            left_file: left_doc.summary.file_name.clone(),
            right_file: right_doc.summary.file_name.clone(),
            text_score,
            image_score,
            page_image_score,
            final_score,
            level: level_for_score(final_score),
            exact_page_match_count: exact_pages.len(),
            approximate_text_match_count,
            matched_text_chars: matched_chars,
            matched_texts,
            matched_images: Vec::new(),
        });
    }

    pairs
}

fn match_exact_pages(
    left_doc: &PdfDocumentData,
    right_doc: &PdfDocumentData,
) -> Vec<(usize, usize)> {
    let mut right_pages_by_hash: HashMap<u64, Vec<usize>> = HashMap::new();
    for (right_index, page) in right_doc.pages.iter().enumerate() {
        if page.char_count >= MIN_EXACT_PAGE_CHARS {
            right_pages_by_hash
                .entry(page.text_hash)
                .or_default()
                .push(right_index);
        }
    }

    let mut matches = Vec::new();
    for (left_index, left_page) in left_doc.pages.iter().enumerate() {
        if left_page.char_count < MIN_EXACT_PAGE_CHARS {
            continue;
        }

        let Some(right_pages) = right_pages_by_hash.get_mut(&left_page.text_hash) else {
            continue;
        };
        let Some(position) = right_pages
            .iter()
            .position(|right_index| right_doc.pages[*right_index].text == left_page.text)
        else {
            continue;
        };
        let right_index = right_pages.swap_remove(position);
        matches.push((left_index, right_index));
    }
    matches.sort_unstable();
    matches
}

fn add_coverage(
    coverage: &mut HashMap<usize, Vec<(usize, usize)>>,
    page: usize,
    start: usize,
    end: usize,
) {
    if end > start {
        coverage.entry(page).or_default().push((start, end));
    }
}

fn covered_chars(coverage: &HashMap<usize, Vec<(usize, usize)>>) -> usize {
    coverage
        .values()
        .map(|ranges| {
            let mut ranges = ranges.clone();
            ranges.sort_unstable();
            ranges
                .into_iter()
                .fold(Vec::<(usize, usize)>::new(), |mut merged, (start, end)| {
                    if let Some((_, previous_end)) = merged.last_mut() {
                        if start <= *previous_end {
                            *previous_end = (*previous_end).max(end);
                            return merged;
                        }
                    }
                    merged.push((start, end));
                    merged
                })
                .into_iter()
                .map(|(start, end)| end - start)
                .sum::<usize>()
        })
        .sum()
}

fn build_groups(pairs: &[SimilarityPair], request: &AnalyzeRequest) -> Vec<SimilarityGroup> {
    let strong_pairs = pairs
        .iter()
        .filter(|pair| pair.final_score >= request.final_threshold)
        .collect::<Vec<_>>();
    let mut adjacency: HashMap<String, HashSet<String>> = HashMap::new();

    for pair in &strong_pairs {
        adjacency
            .entry(pair.left_file.clone())
            .or_default()
            .insert(pair.right_file.clone());
        adjacency
            .entry(pair.right_file.clone())
            .or_default()
            .insert(pair.left_file.clone());
    }

    let mut visited = HashSet::new();
    let mut groups = Vec::new();

    for file in adjacency.keys() {
        if visited.contains(file) {
            continue;
        }

        let mut stack = vec![file.clone()];
        let mut files = Vec::new();
        visited.insert(file.clone());

        while let Some(current) = stack.pop() {
            files.push(current.clone());
            if let Some(neighbors) = adjacency.get(&current) {
                for neighbor in neighbors {
                    if visited.insert(neighbor.clone()) {
                        stack.push(neighbor.clone());
                    }
                }
            }
        }

        files.sort();
        if files.len() < 2 {
            continue;
        }

        let file_set = files.iter().cloned().collect::<HashSet<_>>();
        let relations = strong_pairs
            .iter()
            .filter(|pair| {
                file_set.contains(&pair.left_file) && file_set.contains(&pair.right_file)
            })
            .map(|pair| PairRelation {
                left_file: pair.left_file.clone(),
                right_file: pair.right_file.clone(),
                final_score: pair.final_score,
                text_score: pair.text_score,
                image_score: pair.image_score,
                page_image_score: pair.page_image_score,
            })
            .collect::<Vec<_>>();

        let possible_edges = (files.len() * (files.len() - 1) / 2).max(1);
        let graph_density = relations.len() as f32 / possible_edges as f32;
        let average_score = relations
            .iter()
            .map(|relation| relation.final_score)
            .sum::<f32>()
            / relations.len().max(1) as f32;
        let common_evidence_coverage = graph_density;
        let group_score =
            (average_score * 0.55 + common_evidence_coverage * 0.25 + graph_density * 0.20)
                .min(1.0);
        let mut quality_flags = Vec::new();
        if graph_density < 0.5 {
            quality_flags.push(GroupQualityFlag::WeakConnection);
            quality_flags.push(GroupQualityFlag::NeedsManualReview);
        }

        groups.push(SimilarityGroup {
            group_id: format!("group-{}", groups.len() + 1),
            files,
            group_score,
            level: level_for_score(group_score),
            graph_density,
            quality_flags,
            pair_relations: relations,
            representative_evidence: vec![GroupEvidence {
                evidence_type: "text".to_string(),
                summary: "组内文件存在共享文本 shingle 和确认雷同片段。".to_string(),
            }],
        });
    }

    groups.sort_by(|left, right| {
        right
            .group_score
            .partial_cmp(&left.group_score)
            .unwrap_or(Ordering::Equal)
    });
    groups
}

fn clean_page_texts(raw_pages: Vec<(usize, String)>) -> Vec<PageText> {
    let normalized_lines = raw_pages
        .into_iter()
        .map(|(page, raw_text)| {
            let lines = raw_text
                .lines()
                .map(normalize_text)
                .filter(|line| !line.is_empty())
                .collect::<Vec<_>>();
            (page, lines)
        })
        .collect::<Vec<_>>();

    let mut line_frequency: HashMap<String, usize> = HashMap::new();
    for (_, lines) in &normalized_lines {
        for line in lines.iter().cloned().collect::<HashSet<_>>() {
            *line_frequency.entry(line).or_default() += 1;
        }
    }

    let repeated_line_threshold = (normalized_lines.len() / 20).max(4);
    normalized_lines
        .into_iter()
        .map(|(page, lines)| {
            let text = lines
                .into_iter()
                .filter(|line| {
                    let is_short = line.chars().count() <= 120;
                    let frequency = line_frequency.get(line).copied().unwrap_or_default();
                    !(is_short && frequency >= repeated_line_threshold)
                })
                .collect::<Vec<_>>()
                .join(" ");
            let char_count = text.chars().count();
            PageText {
                page,
                text_hash: hash_value(&text),
                text,
                char_count,
            }
        })
        .collect()
}

fn chunk_text(text: &str, page: usize, request: &AnalyzeRequest) -> Vec<TextChunk> {
    let chars = text.chars().collect::<Vec<_>>();
    if chars.len() < request.min_chunk_chars {
        return Vec::new();
    }

    let mut chunks = Vec::new();
    let mut start = 0usize;
    while start < chars.len() {
        let end = (start + request.target_chunk_chars).min(chars.len());
        let chunk_text = chars[start..end].iter().collect::<String>();
        let shingles = shingles(&chunk_text, request.shingle_size);
        if !shingles.is_empty() {
            let shingle_set = shingles.iter().copied().collect::<HashSet<_>>();
            chunks.push(TextChunk {
                page,
                start,
                end,
                text_hash: hash_value(&chunk_text),
                text: chunk_text,
                simhash: simhash(&shingles),
                shingles,
                shingle_set,
            });
        }
        if end == chars.len() {
            break;
        }
        start = end.saturating_sub(request.chunk_overlap_chars);
    }

    chunks
}

fn shingles(text: &str, shingle_size: usize) -> Vec<u64> {
    let chars = text.chars().collect::<Vec<_>>();
    if chars.len() < shingle_size {
        return Vec::new();
    }

    let mut unique = HashSet::new();
    for window in chars.windows(shingle_size) {
        let token = window.iter().collect::<String>();
        unique.insert(hash_value(&token));
    }

    let mut values = unique.into_iter().collect::<Vec<_>>();
    values.sort_unstable();
    if values.len() > 220 {
        let step = (values.len() / 220).max(1);
        values = values.into_iter().step_by(step).take(220).collect();
    }
    values
}

fn normalize_text(input: &str) -> String {
    let mut output = String::with_capacity(input.len());
    let mut last_space = true;

    for ch in input.chars() {
        let normalized = if ch.is_ascii_alphanumeric() {
            Some(ch.to_ascii_lowercase())
        } else if is_cjk(ch) || is_cid_fingerprint(ch) {
            Some(ch)
        } else if ch.is_whitespace() {
            Some(' ')
        } else {
            None
        };

        if let Some(ch) = normalized {
            if ch == ' ' {
                if !last_space {
                    output.push(ch);
                    last_space = true;
                }
            } else {
                output.push(ch);
                last_space = false;
            }
        }
    }

    output.trim().to_string()
}

fn is_cid_fingerprint(ch: char) -> bool {
    (CID_FINGERPRINT_BASE..=CID_FINGERPRINT_BASE + u16::MAX as u32).contains(&(ch as u32))
}

fn is_cjk(ch: char) -> bool {
    ('\u{4e00}'..='\u{9fff}').contains(&ch)
        || ('\u{3400}'..='\u{4dbf}').contains(&ch)
        || ('\u{f900}'..='\u{faff}').contains(&ch)
}

fn simhash(shingles: &[u64]) -> u64 {
    let mut weights = [0i32; 64];
    for hash in shingles {
        for bit in 0..64 {
            if (hash >> bit) & 1 == 1 {
                weights[bit] += 1;
            } else {
                weights[bit] -= 1;
            }
        }
    }

    weights.iter().enumerate().fold(0u64, |acc, (bit, weight)| {
        if *weight >= 0 {
            acc | (1u64 << bit)
        } else {
            acc
        }
    })
}

fn jaccard_similarity(left: &HashSet<u64>, right: &HashSet<u64>) -> f32 {
    if left.is_empty() || right.is_empty() {
        return 0.0;
    }

    let (small, large) = if left.len() <= right.len() {
        (left, right)
    } else {
        (right, left)
    };
    let intersection = small.iter().filter(|hash| large.contains(hash)).count();
    let union = left.len() + right.len() - intersection;
    intersection as f32 / union.max(1) as f32
}

fn hamming_distance(left: u64, right: u64) -> u32 {
    (left ^ right).count_ones()
}

fn hash_value(value: &str) -> u64 {
    let mut hasher = DefaultHasher::new();
    value.hash(&mut hasher);
    hasher.finish()
}

fn count_images(document: &Document) -> usize {
    document
        .objects
        .values()
        .filter(|object| match object {
            Object::Stream(stream) => {
                matches!(stream.dict.get(b"Subtype"), Ok(Object::Name(name)) if name == b"Image")
            }
            _ => false,
        })
        .count()
}

pub fn export_report(
    result: &AnalysisResult,
    request: &ExportReportRequest,
) -> Result<ExportReportResult, String> {
    if !request.export_json && !request.export_word {
        return Err("请至少选择一种报告格式。".to_string());
    }

    let output_dir = Path::new(&request.target_dir);
    fs::create_dir_all(output_dir).map_err(|error| format!("无法创建导出目录: {error}"))?;
    let task_label = result.task_id.chars().take(8).collect::<String>();
    let base_name = format!("PDF雷同性检测报告_{task_label}");
    let mut exported_files = Vec::new();

    if request.export_word {
        let path = output_dir.join(format!("{base_name}.docx"));
        write_word_report(result, request.include_text_evidence, &path)?;
        exported_files.push(path.to_string_lossy().to_string());
    }

    if request.export_json {
        let path = output_dir.join(format!("{base_name}.json"));
        let json = serde_json::to_string_pretty(result).map_err(|error| error.to_string())?;
        fs::write(&path, json).map_err(|error| format!("JSON 报告写入失败: {error}"))?;
        exported_files.push(path.to_string_lossy().to_string());
    }

    Ok(ExportReportResult { exported_files })
}

fn write_word_report(
    result: &AnalysisResult,
    include_text_evidence: bool,
    path: &Path,
) -> Result<(), String> {
    use zip::write::SimpleFileOptions;
    use zip::{CompressionMethod, ZipWriter};

    let file = fs::File::create(path).map_err(|error| format!("Word 报告创建失败: {error}"))?;
    let mut zip = ZipWriter::new(file);
    let options = SimpleFileOptions::default().compression_method(CompressionMethod::Deflated);
    let document_xml = build_word_document(result, include_text_evidence);
    let files = [
        (
            "[Content_Types].xml",
            r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  <Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
</Types>"#
                .to_string(),
        ),
        (
            "_rels/.rels",
            r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>"#
                .to_string(),
        ),
        (
            "word/styles.xml",
            r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:docDefaults>
    <w:rPrDefault><w:rPr><w:rFonts w:ascii="Arial" w:eastAsia="Microsoft YaHei"/><w:sz w:val="21"/></w:rPr></w:rPrDefault>
  </w:docDefaults>
  <w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/></w:style>
  <w:style w:type="paragraph" w:styleId="Title"><w:name w:val="Title"/><w:basedOn w:val="Normal"/><w:rPr><w:b/><w:sz w:val="34"/><w:color w:val="1F4E79"/></w:rPr></w:style>
  <w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/><w:basedOn w:val="Normal"/><w:rPr><w:b/><w:sz w:val="27"/><w:color w:val="1F4E79"/></w:rPr></w:style>
  <w:style w:type="paragraph" w:styleId="Heading2"><w:name w:val="heading 2"/><w:basedOn w:val="Normal"/><w:rPr><w:b/><w:sz w:val="23"/><w:color w:val="365F91"/></w:rPr></w:style>
</w:styles>"#
                .to_string(),
        ),
        ("word/document.xml", document_xml),
    ];

    for (name, contents) in files {
        zip.start_file(name, options)
            .map_err(|error| format!("Word 报告压缩失败: {error}"))?;
        zip.write_all(contents.as_bytes())
            .map_err(|error| format!("Word 报告写入失败: {error}"))?;
    }
    zip.finish()
        .map_err(|error| format!("Word 报告封装失败: {error}"))?;
    Ok(())
}

fn build_word_document(result: &AnalysisResult, include_text_evidence: bool) -> String {
    let settings = &result.analysis_settings;
    let ready_files = result
        .files
        .iter()
        .filter(|file| file.status == "ready" || file.status == "cid-fallback")
        .count();
    let failed_files = result.files.len().saturating_sub(ready_files);
    let exact_pages = result
        .pairs
        .iter()
        .map(|pair| pair.exact_page_match_count)
        .sum::<usize>();
    let approximate_matches = result
        .pairs
        .iter()
        .map(|pair| pair.approximate_text_match_count)
        .sum::<usize>();
    let mut body = String::new();

    body.push_str(&word_paragraph("PDF 雷同性检测报告", Some("Title")));
    body.push_str(&word_paragraph(
        &format!(
            "任务编号：{}    开始时间：{}    完成时间：{}",
            result.task_id,
            result.started_at,
            result.finished_at.as_deref().unwrap_or("--")
        ),
        None,
    ));
    body.push_str(&word_paragraph(
        "说明：本报告由本地文本检测引擎生成。图片 pHash、扫描件页面渲染和 OCR 尚未启用，结论应结合人工复核使用。",
        None,
    ));

    body.push_str(&word_paragraph("1. 分析摘要", Some("Heading1")));
    body.push_str(&word_table(
        &["指标", "结果", "指标", "结果"],
        &[
            vec![
                "导入文件",
                &result.files.len().to_string(),
                "成功解析",
                &ready_files.to_string(),
            ],
            vec![
                "解析失败",
                &failed_files.to_string(),
                "候选确认关系",
                &result.pairs.len().to_string(),
            ],
            vec![
                "雷同组",
                &result.groups.len().to_string(),
                "精确雷同页",
                &exact_pages.to_string(),
            ],
            vec![
                "近似文本片段",
                &approximate_matches.to_string(),
                "分析深度",
                &settings.analysis_depth,
            ],
        ],
    ));

    body.push_str(&word_paragraph("2. 本次参数", Some("Heading1")));
    body.push_str(&word_table(
        &["参数", "值", "参数", "值"],
        &[
            vec![
                "文本确认阈值",
                &format!("{:.2}", settings.text_threshold),
                "成组阈值",
                &format!("{:.2}", settings.final_threshold),
            ],
            vec![
                "目标块长度",
                &settings.target_chunk_chars.to_string(),
                "块重叠字符",
                &settings.chunk_overlap_chars.to_string(),
            ],
            vec![
                "shingle 粒度",
                &settings.shingle_size.to_string(),
                "最少共享 shingle",
                &settings.min_shared_shingles.to_string(),
            ],
            vec![
                "SimHash 容差",
                &settings.simhash_hamming_threshold.to_string(),
                "每文件候选上限",
                &settings.candidate_top_k_per_file.to_string(),
            ],
            vec![
                "每对证据上限",
                &settings.max_matches_per_pair.to_string(),
                "候选阈值",
                &format!("{:.2}", settings.candidate_score_threshold),
            ],
        ],
    ));

    body.push_str(&word_paragraph("3. 雷同组", Some("Heading1")));
    if result.groups.is_empty() {
        body.push_str(&word_paragraph("未发现达到成组阈值的雷同组。", None));
    }
    for (index, group) in result.groups.iter().enumerate() {
        body.push_str(&word_paragraph(
            &format!(
                "雷同组 {}：{} 个文件，组评分 {:.2}，关系密度 {:.2}",
                index + 1,
                group.files.len(),
                group.group_score,
                group.graph_density
            ),
            Some("Heading2"),
        ));
        body.push_str(&word_paragraph(
            &format!("成员：{}", group.files.join("；")),
            None,
        ));
        let relation_rows = group
            .pair_relations
            .iter()
            .map(|relation| {
                vec![
                    relation.left_file.clone(),
                    relation.right_file.clone(),
                    format!("{:.2}", relation.final_score),
                    format!("{:.2}", relation.text_score),
                ]
            })
            .collect::<Vec<_>>();
        body.push_str(&word_table_owned(
            &["文件 A", "文件 B", "综合", "文本"],
            &relation_rows,
        ));
    }

    body.push_str(&word_paragraph("4. 文件对证据", Some("Heading1")));
    if result.pairs.is_empty() {
        body.push_str(&word_paragraph("未发现确认关系。", None));
    }
    for (index, pair) in result.pairs.iter().enumerate() {
        body.push_str(&word_paragraph(
            &format!(
                "关系 {}：{} ↔ {}",
                index + 1,
                pair.left_file,
                pair.right_file
            ),
            Some("Heading2"),
        ));
        body.push_str(&word_table(
            &["综合分", "精确页", "近似片段", "覆盖字符"],
            &[vec![
                &format!("{:.2}", pair.final_score),
                &pair.exact_page_match_count.to_string(),
                &pair.approximate_text_match_count.to_string(),
                &pair.matched_text_chars.to_string(),
            ]],
        ));
        if include_text_evidence {
            for evidence in pair.matched_texts.iter().take(5) {
                body.push_str(&word_paragraph(
                    &format!(
                        "A 第 {} 页 / B 第 {} 页，相似度 {:.2}：{}",
                        evidence.left_page,
                        evidence.right_page,
                        evidence.similarity,
                        evidence.left_text
                    ),
                    None,
                ));
            }
        }
    }

    if !result.warnings.is_empty() {
        body.push_str(&word_paragraph("5. 提示与限制", Some("Heading1")));
        for warning in &result.warnings {
            body.push_str(&word_paragraph(&format!("• {warning}"), None));
        }
    }

    format!(
        r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>{body}<w:sectPr><w:pgSz w:w="11906" w:h="16838"/><w:pgMar w:top="1134" w:right="1134" w:bottom="1134" w:left="1134"/></w:sectPr></w:body>
</w:document>"#
    )
}

fn word_paragraph(text: &str, style: Option<&str>) -> String {
    let style_xml = style
        .map(|style| {
            format!(
                r#"<w:pPr><w:pStyle w:val="{}"/></w:pPr>"#,
                xml_escape(style)
            )
        })
        .unwrap_or_default();
    format!(
        r#"<w:p>{style_xml}<w:r><w:t xml:space="preserve">{}</w:t></w:r></w:p>"#,
        xml_escape(text)
    )
}

fn word_table(headers: &[&str], rows: &[Vec<&str>]) -> String {
    let mut xml = String::from(
        r#"<w:tbl><w:tblPr><w:tblBorders><w:top w:val="single" w:sz="4" w:color="B7C9DB"/><w:left w:val="single" w:sz="4" w:color="B7C9DB"/><w:bottom w:val="single" w:sz="4" w:color="B7C9DB"/><w:right w:val="single" w:sz="4" w:color="B7C9DB"/><w:insideH w:val="single" w:sz="4" w:color="D9E2F0"/><w:insideV w:val="single" w:sz="4" w:color="D9E2F0"/></w:tblBorders></w:tblPr>"#,
    );
    xml.push_str(&word_table_row(headers, true));
    for row in rows {
        xml.push_str(&word_table_row(row, false));
    }
    xml.push_str("</w:tbl>");
    xml
}

fn word_table_owned(headers: &[&str], rows: &[Vec<String>]) -> String {
    let mut xml = String::from(
        r#"<w:tbl><w:tblPr><w:tblBorders><w:top w:val="single" w:sz="4" w:color="B7C9DB"/><w:left w:val="single" w:sz="4" w:color="B7C9DB"/><w:bottom w:val="single" w:sz="4" w:color="B7C9DB"/><w:right w:val="single" w:sz="4" w:color="B7C9DB"/><w:insideH w:val="single" w:sz="4" w:color="D9E2F0"/><w:insideV w:val="single" w:sz="4" w:color="D9E2F0"/></w:tblBorders></w:tblPr>"#,
    );
    xml.push_str(&word_table_row(headers, true));
    for row in rows {
        let cells = row.iter().map(String::as_str).collect::<Vec<_>>();
        xml.push_str(&word_table_row(&cells, false));
    }
    xml.push_str("</w:tbl>");
    xml
}

fn word_table_row(cells: &[&str], header: bool) -> String {
    let mut xml = String::from("<w:tr>");
    for cell in cells {
        let shade = if header {
            r#"<w:tcPr><w:shd w:fill="D9EAF7"/></w:tcPr>"#
        } else {
            ""
        };
        xml.push_str(&format!(
            r#"<w:tc>{shade}<w:p><w:r>{}<w:t xml:space="preserve">{}</w:t></w:r></w:p></w:tc>"#,
            if header { "<w:rPr><w:b/></w:rPr>" } else { "" },
            xml_escape(cell)
        ));
    }
    xml.push_str("</w:tr>");
    xml
}

fn xml_escape(text: &str) -> String {
    text.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}

fn write_report(result: &AnalysisResult) -> Result<PathBuf, String> {
    let cwd = std::env::current_dir().map_err(|error| error.to_string())?;
    let output_dir = cwd.join("analysis_results");
    fs::create_dir_all(&output_dir).map_err(|error| error.to_string())?;
    let path = output_dir.join(format!("report_{}.json", result.task_id));
    let json = serde_json::to_string_pretty(result).map_err(|error| error.to_string())?;
    fs::write(&path, json).map_err(|error| error.to_string())?;
    Ok(path)
}

fn update_progress(
    progress: &mut AnalysisProgress,
    timer: Instant,
    on_progress: &impl Fn(AnalysisProgress),
) {
    progress.elapsed_seconds = timer.elapsed().as_secs();
    if progress.processed_pages > 0 && progress.total_pages > progress.processed_pages {
        let pages_left = progress.total_pages - progress.processed_pages;
        let seconds_per_page =
            progress.elapsed_seconds as f32 / progress.processed_pages.max(1) as f32;
        progress.estimated_remaining_seconds = Some((pages_left as f32 * seconds_per_page) as u64);
    }
    on_progress(progress.clone());
}

fn file_name(path: &str) -> String {
    Path::new(path)
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or(path)
        .to_string()
}

fn preview(text: &str) -> String {
    if !text_is_readable(text) {
        return "PDF 缺少 ToUnicode 映射，无法还原可读中文；当前仅使用 CID 字形指纹进行雷同性比对。"
            .to_string();
    }

    let mut value = text.chars().take(360).collect::<String>();
    if text.chars().count() > 360 {
        value.push_str("...");
    }
    value
}

fn text_is_readable(text: &str) -> bool {
    !text.chars().any(is_cid_fingerprint)
}

fn level_for_score(score: f32) -> SimilarityLevel {
    if score >= 0.85 {
        SimilarityLevel::Extreme
    } else if score >= 0.70 {
        SimilarityLevel::High
    } else if score >= 0.50 {
        SimilarityLevel::Medium
    } else {
        SimilarityLevel::Low
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalizes_mixed_text() {
        assert_eq!(normalize_text(" A股！深圳 长亮科技 "), "a股深圳 长亮科技");
    }

    #[test]
    fn preserves_identity_h_cids_for_fingerprinting() {
        let fingerprint = identity_h_cids_to_fingerprint(&[0x0a, 0xc5, 0x07, 0x53]);

        assert_eq!(fingerprint.chars().count(), 2);
        assert_eq!(normalize_text(&fingerprint), fingerprint);
        assert!(!text_is_readable(&fingerprint));
        assert!(preview(&fingerprint).contains("无法还原可读中文"));
    }

    #[test]
    fn detects_jaccard_overlap() {
        let left = shingles(
            "深圳市长亮科技股份有限公司软件著作权招标投标文件",
            SHINGLE_SIZE,
        );
        let right = shingles(
            "深圳市长亮科技股份有限公司软件著作权招标投标材料",
            SHINGLE_SIZE,
        );
        let left_set = left.into_iter().collect::<HashSet<_>>();
        let right_set = right.into_iter().collect::<HashSet<_>>();
        assert!(jaccard_similarity(&left_set, &right_set) > 0.45);
    }

    #[test]
    fn filters_weak_single_chunk_candidates() {
        let raw_scores = HashMap::from([((0, 1), 1.0)]);
        let chunk_pairs = HashMap::from([((0, 0, 1, 0), MIN_SHARED_SHINGLES)]);
        let request = AnalyzeRequest::default();

        assert!(select_candidate_pairs(&raw_scores, &chunk_pairs, &[1, 1], &request).is_empty());
    }

    #[test]
    fn keeps_strong_single_chunk_candidates() {
        let raw_scores = HashMap::from([((0, 1), 1.0)]);
        let chunk_pairs = HashMap::from([((0, 0, 1, 0), CANDIDATE_STRONG_SINGLE_CHUNK_SHINGLES)]);
        let request = AnalyzeRequest::default();

        assert!(
            select_candidate_pairs(&raw_scores, &chunk_pairs, &[1, 1], &request)
                .contains_key(&(0, 1))
        );
    }

    #[test]
    fn removes_repeated_short_page_headers() {
        let pages = clean_page_texts(vec![
            (1, "项目投标文件\n第一页正文内容足够长".to_string()),
            (2, "项目投标文件\n第二页正文内容足够长".to_string()),
            (3, "项目投标文件\n第三页正文内容足够长".to_string()),
            (4, "项目投标文件\n第四页正文内容足够长".to_string()),
        ]);

        assert!(pages.iter().all(|page| !page.text.contains("项目投标文件")));
        assert!(pages[0].text.contains("第一页正文内容足够长"));
    }

    #[test]
    fn merges_overlapping_coverage_ranges() {
        let mut coverage = HashMap::new();
        add_coverage(&mut coverage, 1, 0, 100);
        add_coverage(&mut coverage, 1, 80, 160);
        add_coverage(&mut coverage, 2, 10, 30);

        assert_eq!(covered_chars(&coverage), 180);
    }
}
