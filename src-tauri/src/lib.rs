pub mod analysis;

use analysis::{
    AnalysisProgress, AnalysisResult, AnalysisStage, AnalyzeRequest, ExportReportRequest,
    ExportReportResult,
};
use once_cell::sync::Lazy;
use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;
use tauri::Manager;
use uuid::Uuid;

const MAX_RETAINED_COMPLETED_TASKS: usize = 20;

#[derive(Clone)]
struct TaskRecord {
    progress: AnalysisProgress,
    result: Option<AnalysisResult>,
    cancel_requested: bool,
    created_at: Instant,
}

#[derive(Default)]
struct TaskManager {
    tasks: Mutex<HashMap<String, TaskRecord>>,
}

impl TaskManager {
    fn create(&self, task_id: String, total_files: usize) {
        let progress = AnalysisProgress::new(task_id.clone(), total_files);
        self.tasks.lock().insert(
            task_id,
            TaskRecord {
                progress,
                result: None,
                cancel_requested: false,
                created_at: Instant::now(),
            },
        );
    }

    fn update_progress(&self, task_id: &str, progress: AnalysisProgress) {
        if let Some(task) = self.tasks.lock().get_mut(task_id) {
            task.progress = progress;
        }
    }

    fn finish(&self, task_id: &str, result: AnalysisResult) {
        let mut tasks = self.tasks.lock();
        if let Some(task) = tasks.get_mut(task_id) {
            task.progress = AnalysisProgress {
                stage: if task.cancel_requested {
                    AnalysisStage::Cancelled
                } else {
                    AnalysisStage::Done
                },
                message: if task.cancel_requested {
                    "任务已取消，已保留当前结果。".to_string()
                } else {
                    "分析完成，报告已生成。".to_string()
                },
                ..task.progress.clone()
            };
            task.result = Some(result);
        }
        Self::prune_completed(&mut tasks);
    }

    fn fail(&self, task_id: &str, reason: String) {
        let mut tasks = self.tasks.lock();
        if let Some(task) = tasks.get_mut(task_id) {
            task.progress.stage = AnalysisStage::Failed;
            task.progress.message = reason;
        }
        Self::prune_completed(&mut tasks);
    }

    fn cancel(&self, task_id: &str) {
        if let Some(task) = self.tasks.lock().get_mut(task_id) {
            task.cancel_requested = true;
            task.progress.message = "正在取消任务...".to_string();
        }
    }

    fn is_cancelled(&self, task_id: &str) -> bool {
        self.tasks
            .lock()
            .get(task_id)
            .map(|task| task.cancel_requested)
            .unwrap_or(true)
    }

    fn progress(&self, task_id: &str) -> Option<AnalysisProgress> {
        self.tasks
            .lock()
            .get(task_id)
            .map(|task| task.progress.clone())
    }

    fn result(&self, task_id: &str) -> Option<AnalysisResult> {
        self.tasks
            .lock()
            .get(task_id)
            .and_then(|task| task.result.clone())
    }

    fn prune_completed(tasks: &mut HashMap<String, TaskRecord>) {
        let mut completed = tasks
            .iter()
            .filter(|(_, task)| {
                matches!(
                    task.progress.stage,
                    AnalysisStage::Done | AnalysisStage::Cancelled | AnalysisStage::Failed
                )
            })
            .map(|(task_id, task)| (task_id.clone(), task.created_at))
            .collect::<Vec<_>>();
        completed.sort_by_key(|(_, created_at)| *created_at);
        let remove_count = completed.len().saturating_sub(MAX_RETAINED_COMPLETED_TASKS);
        for (task_id, _) in completed.into_iter().take(remove_count) {
            tasks.remove(&task_id);
        }
    }
}

static TASKS: Lazy<Arc<TaskManager>> = Lazy::new(|| Arc::new(TaskManager::default()));

#[tauri::command]
fn create_analysis_task(app: tauri::AppHandle, request: AnalyzeRequest) -> Result<String, String> {
    request.validate()?;
    let report_dir = app
        .path()
        .app_local_data_dir()
        .map_err(|error| format!("无法定位应用数据目录: {error}"))?
        .join("analysis_results");

    let task_id = Uuid::new_v4().to_string();
    TASKS.create(task_id.clone(), request.paths.len());

    let run_task_id = task_id.clone();
    let progress_task_id = task_id.clone();
    let cancel_task_id = task_id.clone();
    tauri::async_runtime::spawn_blocking(move || {
        let result = std::panic::catch_unwind(|| {
            analysis::run_analysis(
                request,
                run_task_id.clone(),
                &report_dir,
                move |progress| TASKS.update_progress(&progress_task_id, progress),
                move || TASKS.is_cancelled(&cancel_task_id),
            )
        });

        match result {
            Ok(result) => TASKS.finish(&run_task_id, result),
            Err(_) => TASKS.fail(&run_task_id, "分析任务异常终止。".to_string()),
        }
    });

    Ok(task_id)
}

#[tauri::command]
fn get_analysis_progress(task_id: String) -> Result<AnalysisProgress, String> {
    TASKS
        .progress(&task_id)
        .ok_or_else(|| format!("任务不存在: {task_id}"))
}

#[tauri::command]
fn cancel_analysis_task(task_id: String) -> Result<(), String> {
    TASKS.cancel(&task_id);
    Ok(())
}

#[tauri::command]
fn get_analysis_result(task_id: String) -> Result<AnalysisResult, String> {
    TASKS
        .result(&task_id)
        .ok_or_else(|| "结果尚未生成。".to_string())
}

#[tauri::command]
fn export_analysis_report(request: ExportReportRequest) -> Result<ExportReportResult, String> {
    let result = TASKS
        .result(&request.task_id)
        .ok_or_else(|| "结果尚未生成，无法导出报告。".to_string())?;
    analysis::export_report(&result, &request)
}

pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .invoke_handler(tauri::generate_handler![
            create_analysis_task,
            get_analysis_progress,
            cancel_analysis_task,
            get_analysis_result,
            export_analysis_report
        ])
        .run(tauri::generate_context!())
        .expect("failed to run Tauri application");
}

#[cfg(test)]
mod tests {
    use super::*;

    fn task_record(task_id: &str, stage: AnalysisStage) -> TaskRecord {
        let mut progress = AnalysisProgress::new(task_id.to_string(), 2);
        progress.stage = stage;
        TaskRecord {
            progress,
            result: None,
            cancel_requested: false,
            created_at: Instant::now(),
        }
    }

    #[test]
    fn prunes_old_completed_tasks_but_keeps_active_tasks() {
        let mut tasks = HashMap::new();
        for index in 0..(MAX_RETAINED_COMPLETED_TASKS + 3) {
            let task_id = format!("done-{index}");
            tasks.insert(task_id.clone(), task_record(&task_id, AnalysisStage::Done));
        }
        tasks.insert(
            "active".to_string(),
            task_record("active", AnalysisStage::ComparingText),
        );

        TaskManager::prune_completed(&mut tasks);

        assert_eq!(tasks.len(), MAX_RETAINED_COMPLETED_TASKS + 1);
        assert!(tasks.contains_key("active"));
    }
}
