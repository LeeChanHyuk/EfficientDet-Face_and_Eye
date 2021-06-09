import cv2

path = "C:\\users\\user\\Desktop\\test\\"

image = cv2.imread(path + "1525046335_1.jpg")
f = open(path + '299.txt')
f.readline()
while(1):
    x, y = f.readline().split(',')
    x = x.replace(" ","")
    y = y.replace(" ","")
    x_num = int(float(x))
    y_num = int(float(y))
    cv2.circle(image,(x_num,y_num),5,(255,0,0),5,cv2.LINE_AA)
    image2 =  cv2.resize(image, dsize=(640,640))
    cv2.namedWindow("test_1")  # create a named window
    cv2.moveWindow("test_1", 40, 30)  # Move it to (40, 30)
    cv2.imshow("test_1",image2)
    cv2.waitKey(1)