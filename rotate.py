from PIL import Image

im = Image.open('./raw_data/Unity_out/Australia/----------OO--QPY--CS-KCE0UC---K------9AL--H.jpg')
im.show()
im = im.rotate(10)
im.show()