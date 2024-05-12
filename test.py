from PIL import ImageFont, ImageDraw, Image
import numpy as np
from easyocr import Reader
import cv2


img = cv2.imread('img4.jpg')
img = cv2.resize(img, (1000, 600))
fontpath = "./arial.ttf"
font = ImageFont.truetype(fontpath, 32)
b,g,r,a = 0,225,0,0
grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(grayscale, (5, 5), 0)
edged = cv2.Canny(blurred, 10, 200)
cv2.waitKey(0)
cv2.destroyAllWindows()

contours, _ = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key = cv2.contourArea, reverse = True)[:5]

for c in contours:
    perimeter = cv2.arcLength(c, True)
    approximation = cv2.approxPolyDP(c, 0.02 * perimeter, True)
    print(approximation)
    if len(approximation) == 4: 
        number_plate_shape = approximation
        break

(x, y, w, h) = cv2.boundingRect(number_plate_shape)
number_plate = grayscale[y:y + h, x:x + w]

reader = Reader(['en'])
detection = reader.readtext(number_plate)

if len(detection) == 0:
    text = "Không có biển số xe"
    img_pil = Image.fromarray(img) 
    draw = ImageDraw.Draw(img_pil)
    draw.text((150, 500), text, font = font, fill = (b, g, r, a))
    img = np.array(img_pil)
    cv2.waitKey(0)
else:
    cv2.drawContours(img, [number_plate_shape], -1, (0, 0, 225), 3)
    text ="Biển số xe: " + f"{detection[0][1]}"
    img_pil = Image.fromarray(img) 
    draw = ImageDraw.Draw(img_pil)
    draw.text((200, 500), text, font = font, fill = (b, g, r, a))
    img = np.array(img_pil)
    cv2.imshow('Plate Detection', img)
    cv2.waitKey(0)