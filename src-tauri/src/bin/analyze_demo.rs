use pdf_similarity_detector_lib::analysis::{
    export_report, run_analysis, AnalyzeRequest, ExportReportRequest,
};
use std::path::{Path, PathBuf};

fn main() {
    let input = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "../demo_files".to_string());
    let mut paths = collect_pdf_paths(Path::new(&input));
    paths.sort();

    if paths.len() < 2 {
        eprintln!("需要至少 2 个 PDF，当前找到 {} 个。", paths.len());
        std::process::exit(1);
    }

    println!("开始分析 {} 个 PDF:", paths.len());
    for path in &paths {
        println!("  - {}", path.display());
    }

    let request = AnalyzeRequest {
        paths: paths
            .into_iter()
            .map(|path| path.to_string_lossy().to_string())
            .collect(),
        analysis_depth: "deep".to_string(),
        candidate_top_k_per_file: 36,
        max_matches_per_pair: 60,
        ..AnalyzeRequest::default()
    };

    let result = run_analysis(
        request,
        "demo".to_string(),
        |progress| {
            println!(
                "[{:?}] {} | {}/{} 页 | 候选 {} | 确认 {}",
                progress.stage,
                progress.message,
                progress.processed_pages,
                progress.total_pages,
                progress.candidate_pairs,
                progress.confirmed_pairs
            );
        },
        || false,
    );

    println!(
        "完成：{} 个文件，{} 个文件对，{} 个雷同组",
        result.files.len(),
        result.pairs.len(),
        result.groups.len()
    );
    if let Some(path) = &result.report_path {
        println!("报告：{path}");
    }
    let export = export_report(
        &result,
        &ExportReportRequest {
            task_id: result.task_id.clone(),
            target_dir: "analysis_results".to_string(),
            export_json: true,
            export_word: true,
            include_text_evidence: true,
        },
    )
    .unwrap_or_else(|error| {
        eprintln!("导出失败：{error}");
        std::process::exit(1);
    });
    for path in export.exported_files {
        println!("导出：{path}");
    }
}

fn collect_pdf_paths(input: &Path) -> Vec<PathBuf> {
    if input.is_file() {
        return vec![input.to_path_buf()];
    }

    std::fs::read_dir(input)
        .map(|entries| {
            entries
                .filter_map(Result::ok)
                .map(|entry| entry.path())
                .filter(|path| {
                    path.extension()
                        .and_then(|ext| ext.to_str())
                        .map(|ext| ext.eq_ignore_ascii_case("pdf"))
                        .unwrap_or(false)
                })
                .collect()
        })
        .unwrap_or_default()
}
