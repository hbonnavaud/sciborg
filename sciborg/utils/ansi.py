class Ansi:

    @staticmethod
    def rgb(r, g, b):
        return f"\u001b[38;2;{r};{g};{b}m"

    @staticmethod
    def bg_rgb(r, g, b):
        return f"\u001b[48;2;{r};{g};{b}m"

    # Regular Colors
    red = "\u001b[31m"
    green = '\033[92m'
    green_deep = "\u001b[32m"
    yellow = '\033[93m'
    yellow_deep = "\u001b[33m"
    blue = "\033[0;94m"
    blue_deep = "\u001b[34m"
    cyan = '\033[96m'
    cyan_deep = "\u001b[36m"
    pink = '\033[95m'
    magenta_deep = "\u001b[35m"
    purple_deep = "\033[0;35m"
    grey = "\u001b[37m"
    black = "\u001b[30m"
    black_deep = "\033[0;90m"
    white = "\033[0;97m"
    white_deep = "\033[0;37m"

    # Bold
    bold = '\033[1m'
    bold_black = "\033[1;30m"
    bold_red = "\033[1;31m"
    bold_green = "\033[1;32m"
    bold_yellow = "\033[1;33m"
    bold_blue = "\033[1;34m"
    bold_purple = "\033[1;35m"
    bold_cyan = "\033[1;36m"
    bold_white = "\033[1;37m"

    # Underline
    underlined = '\033[4m'
    underlined_black = "\033[4;30m"
    underlined_red = "\033[4;31m"
    underlined_green = "\033[4;32m"
    underlined_yellow = "\033[4;33m"
    underlined_blue = "\033[4;34m"
    underlined_purple = "\033[4;35m"
    underlined_cyan = "\033[4;36m"
    underlined_white = "\033[4;37m"

    # Background
    bg_black = "\u001b[40m"
    bg_red = "\u001b[41m"
    bg_green = "\u001b[42m"
    bg_yellow = "\u001b[43m"
    bg_blue = "\u001b[44m"
    bg_magenta = "\u001b[45m"
    bg_cyan = "\u001b[46m"
    bg_white = "\u001b[47m"

    # Bold High Intensity
    bold_high_black = "\033[1;90m"
    bold_high_red = "\033[1;91m"
    bold_high_green = "\033[1;92m"
    bold_high_yellow = "\033[1;93m"
    bold_high_blue = "\033[1;94m"
    bold_high_purple = "\033[1;95m"
    bold_high_cyan = "\033[1;96m"
    bold_high_white = "\033[1;97m"

    # High Intensity backgrounds
    bg_high_black = "\033[0;100m"
    bg_high_red = "\033[0;101m"
    bg_high_green = "\033[0;102m"
    bg_high_yellow = "\033[0;103m"
    bg_high_blue = "\033[0;104m"
    bg_high_purple = "\033[0;105m"
    bg_high_cyan = "\033[0;106m"
    bg_high_white = "\033[0;107m"

    NONE = '\033[0m'
    FAIL = '\033[91m'  # looks like bold + red

    box_top_left = "╭"
    box_top_right = "╮"
    box_bottom_left = "╰"
    box_bottom_right = "╯"
    box_vertical_bar = "│"
    box_horizontal_bar = "─"
    box_array_inter_center = "┼"
    box_array_inter_top = "┬"
    box_array_inter_bottom = "┴"
    box_array_inter_left = "├"
    box_array_inter_right = "┤"


def test_ansi():

    for k in dir(Ansi):
        v = Ansi().__getattribute__(k)
        if not hasattr(v, "__call__") and not k.startswith("__"):
            print(k, ":" + v + " test" + Ansi.NONE, sep="")