# 常见问题

安装与配置过程中常见问题与快速解决方法。

---

## pip 找不到 evotoolkit

- 确保网络可用并升级 pip：`python -m pip install -U pip`。
- 确认包名正确：`evotoolkit`。

---

## ImportError: cannot import name 'Solution'

- 检查版本：`python -c "import evotoolkit, sys; print(evotoolkit.__version__)"`。
- 若存在多个 Python 版本，请使用与安装环境一致的解释器运行代码。

---

## 未检测到 CUDA

- 安装与系统和显卡匹配的 CUDA 工具包与驱动。
- 某些任务需要 `cuda_engineering` 附加依赖：`pip install evotoolkit[cuda_engineering]`。

---

## LLM API 报错（401/403/SSLError）

- 检查并正确配置 API Key。
- 检查系统时间和证书配置。
- 如使用代理，请确保允许访问对应 API 端点的 HTTPS 流量。

