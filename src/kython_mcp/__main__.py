from .server import main, main_http


if __name__ == "__main__":
    import sys

    print("Starting kython_mcp server...")
    if "--http" in sys.argv:
        main_http()
    else:
        main()
