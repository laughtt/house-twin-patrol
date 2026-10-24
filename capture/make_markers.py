import json
from pathlib import Path

import cv2 as cv
from PIL import Image, ImageDraw, ImageFont

DPI = 300
CM = DPI / 2.54
OUT = Path(__file__).parent.parent / "out/print"
DICT = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_50)
ROOMS = {0: ("Kitchen", "beside the door to the living room"), 1: ("Kitchen", "far wall (optional)"),
         2: ("Spare", "long room or corner"), 3: ("Living room", "beside the kitchen door"),
         4: ("Living room", "beside the bedroom door"), 5: ("Living room", "far wall (optional)"),
         6: ("Bedroom", "beside the door to the living room"), 7: ("Bedroom", "far wall (optional)"),
         8: ("Spare", "long room or corner"), 9: ("Spare", "long room or corner")}
DOORS = {20: "Kitchen | Living, face toward KITCHEN", 21: "Kitchen | Living, face toward LIVING",
         22: "Living | Bedroom, face toward LIVING", 23: "Living | Bedroom, face toward BEDROOM"}
BOARD = {"squares_x": 7, "squares_y": 5, "square_m": 0.035, "marker_m": 0.026, "dict": "DICT_5X5_100"}
font = lambda size: ImageFont.load_default(size=size)


def scale_bar(draw, x, y):
    draw.rectangle([x, y, x + 10 * CM, y + 0.4 * CM], fill="black")
    draw.text((x + 10.4 * CM, y - 0.1 * CM), "= 10 cm? If not, print at 100%", fill="black", font=font(40))


def marker(id_, cm):
    return Image.fromarray(cv.aruco.generateImageMarker(DICT, id_, 6 * round(cm * CM / 6)))


def marker_page(id_, title, note, cm=15, inches=(8.5, 11)):
    page = Image.new("L", (int(inches[0] * DPI), int(inches[1] * DPI)), 255)
    img = marker(id_, cm)
    side = img.width
    x = (page.width - side) // 2
    page.paste(img, (x, int(3 * CM)))
    d = ImageDraw.Draw(page)
    y = int(3 * CM) + side + int(1.2 * CM)
    d.text((x, y), f"#{id_}  {title}", fill="black", font=font(110))
    d.text((x, y + 1.6 * CM), note, fill="black", font=font(40))
    scale_bar(d, x, page.height - 3 * CM)
    return page


def two_page(items):
    page = Image.new("L", (int(8.5 * DPI), int(11 * DPI)), 255)
    d = ImageDraw.Draw(page)
    for (id_, text), y in zip(items, (2 * CM, 14.5 * CM)):
        page.paste(marker(id_, 10), (int(2 * CM), int(y)))
        d.text((13.5 * CM, y), f"#{id_}", fill="black", font=font(150))
        d.multiline_text((13.5 * CM, y + 2.2 * CM), text, fill="black", font=font(46), spacing=14)
    scale_bar(d, 2 * CM, page.height - 2 * CM)
    return page


def board_page():
    page = Image.new("L", (int(11 * DPI), int(8.5 * DPI)), 255)
    sq = round(BOARD["square_m"] * 100 * CM)
    board = cv.aruco.CharucoBoard((BOARD["squares_x"], BOARD["squares_y"]), BOARD["square_m"], BOARD["marker_m"],
                                  cv.aruco.getPredefinedDictionary(cv.aruco.DICT_5X5_100))
    img = board.generateImage((BOARD["squares_x"] * sq, BOARD["squares_y"] * sq), marginSize=0)
    x = (page.width - img.shape[1]) // 2
    page.paste(Image.fromarray(img), (x, int(1.0 * CM)))
    d = ImageDraw.Draw(page)
    y = int(1.0 * CM) + img.shape[0] + int(0.5 * CM)
    d.text((x, y), "Lens check board: lay flat on a table. Each square must measure 3.5 cm.", fill="black", font=font(44))
    scale_bar(d, x, y + 1.1 * CM)
    return page


OUT.mkdir(parents=True, exist_ok=True)
note = "Black square = 15 cm. Glue flat on the wall, centre about 20 cm above the floor."
pages = [marker_page(i, ROOMS[i][0], f"{ROOMS[i][1].capitalize()}. {note}") for i in range(10)]
pages.append(marker_page(40, "BLOCKED", "Tape to a box and put it in a doorway to block that route."))
pages.append(board_page())
pages[0].save(OUT / "markers.pdf", save_all=True, append_images=pages[1:], resolution=DPI)
(OUT / "charuco.json").write_text(json.dumps(BOARD, indent=2))

temp = "TEMPORARY: filming only.\nTape on any wall, about\n20-60 cm high. Remove\nafter filming.\nBlack square = 10 cm."
box = "\nGlue centred on one box face;\nits partner goes centred on\nthe opposite face. Stand the\nbox in the doorway while\nfilming. Black square = 10 cm."
items = [(i, temp) for i in range(10, 20)] + [(i, DOORS[i].replace(", ", "\n") + box) for i in DOORS]
filming = [two_page(items[i:i + 2]) for i in range(0, len(items), 2)]
filming[0].save(OUT / "filming.pdf", save_all=True, append_images=filming[1:], resolution=DPI)

big = [marker_page(30 + i, room, "Black square = 20 cm. Optional far-wall marker (longer range). Print on Tabloid or A3 at 100%. "
                   "Glue flat, centre about 25 cm above the floor.", cm=20, inches=(11, 17))
       for i, room in enumerate(["Kitchen far wall", "Living far wall", "Bedroom far wall"])]
big[0].save(OUT / "big.pdf", save_all=True, append_images=big[1:], resolution=DPI)
print(OUT)
