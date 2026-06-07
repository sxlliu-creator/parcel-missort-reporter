"""
WCS 模拟端点启动脚本

运行：  python run_wcs_mock.py
"""
import uvicorn
from tests.wcs_mock import app


def main():
    import sys
    port = 9900

    for arg in sys.argv[1:]:
        if arg.startswith("--port="):
            port = int(arg.split("=", 1)[1])

    print(f"WCS 模拟端点启动: http://0.0.0.0:{port}")
    print(f"查看事件: curl http://localhost:{port}/api/events")
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
