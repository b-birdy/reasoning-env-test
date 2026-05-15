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

    # ── 2. 软件检测 ────────────────────────────────────────────────
    from reasoning_env_test.detectors.software.software import detect_all as detect_software
    software, err = _safe_call(detect_software, "软件检测", verbose)
    if err:
        errors.append(err)
        software = {}

    # ── 3. 容器检测 ────────────────────────────────────────────────
    from reasoning_env_test.detectors.container.container import detect_all as detect_container
    container, err = _safe_call(detect_container, "容器检测", verbose)
    if err:
        errors.append(err)
        container = {}

    # ── 4. 模型推荐 ────────────────────────────────────────────────
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

    # ── 5. 性能预估 ────────────────────────────────────────────────
    performance: List[Dict[str, Any]] = []
    if isinstance(recommendations, list) and recommendations:
        _log("正在性能预估…", verbose)
        try:
            from reasoning_env_test.performance.estimator import estimate_performance
            best = recommendations[0]
            model_params_b: float = best["model"]["params_b"]
            precision: str = best["precision"]
            concurrency_levels = [4, 8, 16, 32, 64, 128]
            performance = estimate_performance(
                hardware, model_params_b, precision, concurrency_levels,
            )
            _log("性能预估完成", verbose)
        except Exception as exc:
            err_msg = f"性能预估失败: {exc}"
            errors.append(err_msg)
            _log(err_msg, verbose)
            if verbose:
                traceback.print_exc(file=sys.stderr)
            performance = []
    else:
        _log("无推荐模型，跳过性能预估", verbose)

    # ── 6. 生成报告 ────────────────────────────────────────────────
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
        )
        _log("报告生成完成", verbose)
    except Exception as exc:
        err_msg = f"报告生成失败: {exc}"
        errors.append(err_msg)
        _log(err_msg, verbose)
        if verbose:
            traceback.print_exc(file=sys.stderr)

    # ── 7. 输出 ────────────────────────────────────────────────────
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

    # ── 8. 总结 ────────────────────────────────────────────────────
    if errors:
        print(
            "\n---\n⚠️ 检测过程中出现以下错误:\n" + "\n".join(f"  - {e}" for e in errors),
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
