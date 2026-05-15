"""推理环境检测工具 — CLI 入口。

Usage:
    python -m reasoning_env_test
    python -m reasoning_env_test --output report.md
    python -m reasoning_env_test --verbose
"""

import argparse
import sys
import traceback
from typing import Any, Dict, List, Union


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="推理环境检测工具 — 检测硬件/软件/容器环境，推荐可部署模型并预估性能",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例:\n"
            "  python -m reasoning_env_test\n"
            "  python -m reasoning_env_test -o report.md\n"
            "  python -m reasoning_env_test -v\n"
        ),
    )
    parser.add_argument(
        "-o", "--output",
        metavar="FILE",
        help="输出到指定文件（默认输出到 stdout）",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="输出详细执行信息",
    )
    return parser.parse_args(argv)


def _log(msg: str, verbose: bool) -> None:
    """根据 verbose 模式打印日志。"""
    if verbose:
        print(f"[INFO] {msg}", file=sys.stderr)


def _write_utf8(fh, content: str) -> None:
    """以 UTF-8 编码写入文件句柄，避免 Windows GBK 编码问题。"""
    try:
        fh.buffer.write(content.encode("utf-8"))
        fh.buffer.flush()
    except AttributeError:
        # 没有 buffer 属性（如测试 mock）时降级
        fh.write(content)


def _safe_call(
    fn, args_desc: str, verbose: bool,
) -> tuple[Any, str | None]:
    """安全调用目标函数，返回 (result, error_msg)。"""
    _log(f"正在{args_desc}…", verbose)
    try:
        result = fn()
        _log(f"{args_desc}完成", verbose)
        return result, None
    except Exception as exc:
        err_msg = f"{args_desc}失败: {exc}"
        _log(err_msg, verbose)
        if verbose:
            traceback.print_exc(file=sys.stderr)
        return None, err_msg


def main(argv: List[str] | None = None) -> int:
    """主入口。

    Returns:
        0 表示成功，1 表示部分失败。
    """
    args = parse_args(argv)
    verbose = args.verbose
    errors: List[str] = []

    # ── 1. 硬件检测 ────────────────────────────────────────────────
    from reasoning_env_test.detectors.hardware import detect_all as detect_hardware
    hardware, err = _safe_call(detect_hardware, "硬件检测", verbose)
    if err:
        errors.append(err)
        hardware = []

    # ── 2. 系统信息检测 ─────────────────────────────────────────────
    from reasoning_env_test.detectors.system.system_info import detect_all as detect_system_info
    system_info, err = _safe_call(detect_system_info, "系统信息检测", verbose)
    if err:
        errors.append(err)
        system_info = {}

    # ── 3. 网络检测 ─────────────────────────────────────────────────
    from reasoning_env_test.detectors.network.network import detect_all as detect_network
    network, err = _safe_call(detect_network, "网络检测", verbose)
    if err:
        errors.append(err)
        network = {}

    # ── 4. 软件检测 ────────────────────────────────────────────────
    from reasoning_env_test.detectors.software.software import detect_all as detect_software
    software, err = _safe_call(detect_software, "软件检测", verbose)
    if err:
        errors.append(err)
        software = {}

    # ── 5. 容器检测 ────────────────────────────────────────────────
    from reasoning_env_test.detectors.container.container import detect_all as detect_container
    container, err = _safe_call(detect_container, "容器检测", verbose)
    if err:
        errors.append(err)
        container = {}

    # ── 6. 模型推荐 ────────────────────────────────────────────────
    recommendations: Union[List[Dict[str, Any]], Dict[str, Any]] = {}
    if hardware:
        _log("正在模型推荐…", verbose)
        try:
            from reasoning_env_test.recommend.recommender import ModelRecommender
            recommender = ModelRecommender(hardware)
            recommendations = recommender.recommend()
            _log("模型推荐完成", verbose)
        except Exception as exc:
            err_msg = f"模型推荐失败: {exc}"
            errors.append(err_msg)
            _log(err_msg, verbose)
            if verbose:
                traceback.print_exc(file=sys.stderr)
            recommendations = {"error": True, "message": err_msg}
    else:
        recommendations = {
            "error": True,
            "message": "硬件检测无结果，无法进行模型推荐",
        }
        _log("硬件检测无结果，跳过模型推荐", verbose)

    # ── 7. 性能预估 ────────────────────────────────────────────────
    performance: List[Dict[str, Any]] = []
    _log("正在性能预估…", verbose)
    try:
        from reasoning_env_test.performance.estimator import estimate_performance, load_gpu_specs

        # 统计 GPU 卡数和型号
        gpu_count = 0
        gpu_type_id = "nvidia_a100_80g"  # 默认
        has_rdma = False

        for hw in (hardware or []):
            if hw.get("type") in ("nvidia", "amd", "ascend", "kunlunxin", "hygon") and hw.get("memory_total_gb", 0) > 0:
                details = hw.get("details", {})
                gpu_count += details.get("gpu_count", 0) or details.get("xpu_count", 0) or 1
                if hw["type"] == "kunlunxin":
                    gpu_type_id = "kunlunxin_p800"
                elif hw["type"] == "ascend":
                    gpu_type_id = "ascend_910b"
                elif hw["type"] == "hygon":
                    gpu_type_id = "hygon_dcu_z100"
                elif hw["type"] == "nvidia":
                    gpu_type_id = "nvidia_a100_80g"

        if network and network.get("rdma_devices"):
            has_rdma = len(network["rdma_devices"]) > 0

        gpu_specs = load_gpu_specs()
        gpu_spec = gpu_specs.get(gpu_type_id, gpu_specs["kunlunxin_p800"])

        # 递进展示所有符合条件的场景（非互斥）
        all_perf: List[Dict[str, Any]] = []

        # 场景一：单机单卡推理（始终展示）
        _log("生成场景一：单机单卡推理 (Qwen3.6-27B)", verbose)
        all_perf.extend(estimate_performance(
            gpu_spec, 27.0, "fp16", [8, 16, 32], 1,
            scenario="single_card",
            model_name="Qwen3.6-27B",
            deploy_desc="单机单卡推理"
        ))

        # 场景二：单机多卡分布式推理（gpu_count >= 8 时展示）
        if gpu_count >= 8:
            _log(f"检测到 {gpu_count} 卡 → 生成场景二：单机多卡 (DeepSeek-V3.2)", verbose)
            all_perf.extend(estimate_performance(
                gpu_spec, 685.0, "int8", [32, 64, 128], gpu_count,
                scenario="single_node_multi",
                model_name="DeepSeek-V3.2 (685B)",
                deploy_desc=f"单机 {gpu_count} 卡分布式推理"
            ))

        # 场景三：多机多卡分布式推理（gpu_count >= 8 且有 RDMA 时展示）
        if gpu_count >= 8 and has_rdma:
            _log(f"检测到 {gpu_count} 卡 + RDMA → 生成场景三：多机多卡 (GLM-5.1)", verbose)
            all_perf.extend(estimate_performance(
                gpu_spec, 754.0, "int4", [64, 128, 256], gpu_count * 2,
                memory_bw_factor=0.85, rdma_latency_us=5.0,
                scenario="multi_node",
                model_name="GLM-5.1 (754B)",
                deploy_desc=f"双机 {gpu_count * 2} 卡 RDMA 分布式推理"
            ))

        performance = all_perf
        _log(f"性能预估完成 ({len(performance)} 条)", verbose)
    except Exception as exc:
        err_msg = f"性能预估失败: {exc}"
        errors.append(err_msg)
        _log(err_msg, verbose)
        if verbose:
            traceback.print_exc(file=sys.stderr)
        performance = []

    # ── 8. 生成报告 ────────────────────────────────────────────────
    report_content: str = ""
    _log("正在生成报告…", verbose)
    try:
        from reasoning_env_test.report.reporter import generate_report
        report_content = generate_report(
            hardware=hardware or [],
            software=software or {},
            container=container or {},
            recommendations=recommendations or {},
            performance=performance or [],
            system_info=system_info or {},
            network=network or {},
        )
        _log("报告生成完成", verbose)
    except Exception as exc:
        err_msg = f"报告生成失败: {exc}"
        errors.append(err_msg)
        _log(err_msg, verbose)
        if verbose:
            traceback.print_exc(file=sys.stderr)

    # ── 9. 输出 ────────────────────────────────────────────────────
    if args.output:
        try:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(report_content)
            _log(f"报告已写入: {args.output}", verbose)
        except OSError as exc:
            err_msg = f"写入输出文件失败: {exc}"
            errors.append(err_msg)
            _log(err_msg, verbose)
            # 回退到 stdout
            _write_utf8(sys.stdout, report_content)
    else:
        _write_utf8(sys.stdout, report_content)

    # ── 10. 总结 ───────────────────────────────────────────────────
    if errors:
        print(
            "\n---\n⚠️ 检测过程中出现以下错误:\n" + "\n".join(f"  - {e}" for e in errors),
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
