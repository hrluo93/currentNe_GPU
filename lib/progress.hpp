
#pragma once
#include <mutex>
#include <string>
#include <fstream>
#include <sstream>

class ProgressStatus {
public:
    ProgressStatus() : total_steps_(0), cur_(0), enabled_(false) {}
    ProgressStatus(const ProgressStatus&) = delete;
    ProgressStatus& operator=(const ProgressStatus&) = delete;

    void InitTotalTasks(int totalTasks, const char* filepath) {
        std::lock_guard<std::mutex> lk(mu_);
        total_steps_ = totalTasks;
        cur_ = 0;
        path_ = filepath ? std::string(filepath) : std::string();
        enabled_ = !path_.empty();
        if (enabled_) write("task: 0\nprogress: 0/0\n");
    }

    void InitCurrentTask(int total, const char* msg = nullptr) {
        std::lock_guard<std::mutex> lk(mu_);
        total_steps_ = total > 0 ? total : 0;
        cur_ = 0;
        if (enabled_) {
            std::ostringstream oss;
            if (msg) oss << "task: " << msg << "\n";
            oss << "progress: 0/" << total_steps_ << "\n";
            write(oss.str());
        }
    }

    void SetCurrentTask(int total, const char* msg = nullptr) {
        InitCurrentTask(total, msg);
    }

    void SetTaskProgress(int cur) {
        std::lock_guard<std::mutex> lk(mu_);
        cur_ = cur;
        if (cur_ < 0) cur_ = 0;
        if (cur_ > total_steps_) cur_ = total_steps_;
        if (enabled_) {
            std::ostringstream oss;
            oss << "progress: " << cur_ << "/" << total_steps_ << "\n";
            append(oss.str());
        }
    }

    void SaveProgress() {
        std::lock_guard<std::mutex> lk(mu_);
        if (enabled_) {
            std::ostringstream oss;
            oss << "progress: " << cur_ << "/" << total_steps_ << "\n";
            append(oss.str());
        }
    }

private:
    void write(const std::string& s) {
        std::ofstream ofs(path_, std::ios::out | std::ios::trunc);
        if (ofs.good()) ofs << s;
    }
    void append(const std::string& s) {
        std::ofstream ofs(path_, std::ios::out | std::ios::app);
        if (ofs.good()) ofs << s;
    }

    std::mutex mu_;
    std::string path_;
    int total_steps_;
    int cur_;
    bool enabled_;
};
