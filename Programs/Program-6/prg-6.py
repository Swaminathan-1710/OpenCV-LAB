import cv2
img=cv2.imread('Fortuner.jpg')
text='Fortuner :)'
position=(8,24)
font=cv2.FONT_HERSHEY_SIMPLEX
font_scale=1
color=(255,0,0)
thickness=2
cv2.putText(img,text,position,font,font_scale,color,thickness)
cv2.imshow("Image With Text",img)
cv2.waitKey(0)
cv2.destroyAllWindows()