from PIL import Image
import matplotlib.pyplot as plt
from time import sleep
from robo import path_pre as path
def extent(obj_img, obj_pos):
  xl,yl = obj_img.size
  r1,r0 = obj_pos
  return (r0*5, r0*5 + xl, r1*5+yl, r1*5)

img = Image.open("../assets/Map2.png")
robimg = Image.open("../assets/robot.png")
pickimg = Image.open("../assets/pick.png")
dropimg = Image.open("../assets/drop.png")
for state in path :
  plt.imshow(img)
  rp = state[0][0]
  pps = state[0][1]
  dps = state[0][2]
  for i in pps:
    plt.imshow(pickimg, extent=extent(pickimg, i))
  for i in dps:
    plt.imshow(dropimg, extent=extent(dropimg, i))
  plt.imshow(robimg, extent=extent(robimg, rp))
  plt.imshow(img, alpha=0)
  plt.pause(1/4)

plt.show()