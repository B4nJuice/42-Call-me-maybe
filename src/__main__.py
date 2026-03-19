def main() -> None:
    from .IO.IOManager import IOManager

    io_man: IOManager = IOManager()
    io_man.parse_args()
    print(io_man.args_config)
    print(io_man.get_input())


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(e)
