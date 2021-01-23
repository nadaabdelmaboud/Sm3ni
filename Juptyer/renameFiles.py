import os
print(os.listdir("."))
for dirname in os.listdir(".")[-1]:
    if os.path.isdir(dirname):
        for i, filename in enumerate(os.listdir(dirname)):
            os.rename(dirname + "/" + filename, dirname + "/" + str(i) + ".bmp")

print(os.listdir("."))