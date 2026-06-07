"""
汇聚服务启动脚本

运行：  python run_hub.py
"""
import uvicorn
from src.hub.server import app, set_wcs_endpoint


def main():
    import sys
    port = 8800
    wcs_url = ""

    for i, arg in enumerate(sys.argv[1:], 1):
        if arg.startswith("--port="):
            port = int(arg.split("=", 1)[1])
        elif arg.startswith("--wcs="):
            wcs_url = arg.split("=", 1)[1]

    if wcs_url:
        set_wcs_endpoint(wcs_url)
        print(f"WCS 转发地址: {wcs_url}")

    print(f"中央汇聚服务启动: http://0.0.0.0:{port}")
    print(f"监控面板: http://localhost:{port}/docs")
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
