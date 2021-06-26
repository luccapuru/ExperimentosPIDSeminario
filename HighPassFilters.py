import cv2 
import numpy as np 

class HPFilters():
    def HP1(self, img, kernel):
        img2 = np.ones((img.shape[0],img.shape[1],1),np.uint8)*255
        img2 = cv2.filter2D(img, -1, kernel)
        return img2
    
    def FourierHP(self, img):
        dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        magnitude_spectrum = 20*np.log((cv2.magnitude(dft_shift[:,:,0], dft_shift[:,:,1 ]))+1) 
        rows, cols = img.shape
        crow, ccol = int(rows/2), int(cols/2)
        mask = np.ones((rows,cols,2), np.uint8)
        r = 80
        center =[crow,ccol]
        x, y = np.ogrid[:rows, :cols]
        mask_area = (x - center[0])**2 + (y - center[1]) **2 <= r*r #eq da circunferencia
        mask[mask_area] = 0
        # cv2.imshow("Mascara",mask*255)
        fshift = dft_shift*mask
        fshift_mask_mag = 20*np.log(cv2.magnitude(fshift[:,:,0], fshift[:,:,1]))
        f_ishift = np.fft.ifftshift(fshift)
        imgfiltrada = cv2.idft(f_ishift)
        imgfiltrada = cv2.magnitude(imgfiltrada[:,:,0], imgfiltrada[:,:,1])
        return imgfiltrada

def main():
    hpf = HPFilters()
    #abrindo imagem em escala de cinza
    img = cv2.imread("images//all_souls_000005.jpg", cv2.IMREAD_GRAYSCALE)
    #mostrando a imagem orginal 
    cv2.imshow("Imagem Original", img)
    sharpen_kernel = np.array([[-1,-1,-1], 
                               [-1,9,-1], 
                               [-1,-1,-1]])
    imgHP = hpf.HP1(img, sharpen_kernel)

    sharpen_kernel = np.array([[-1,-1,-1], 
                               [-1,8,-1], 
                               [-1,-1,-1]])
    imgHP2 = hpf.HP1(img, sharpen_kernel)

    sharpen_kernel = np.array([[0,-1/4,0], 
                               [-1/4,2,-1/4], 
                               [0,-1,0]])
    imgHP3 = hpf.HP1(img, sharpen_kernel)

    sharpen_kernel = np.array([[-1,-4,-1],
                               [-4,20,-4],
                               [-1,-4,-1]])
    imgHP4 = hpf.HP1(img, sharpen_kernel)

    imgFHP = hpf.FourierHP(img)
    
    cv2.imshow("Imagem - HP1", imgHP)
    cv2.imshow("Imagem - HP2", imgHP2)
    cv2.imshow("Imagem - HP3", imgHP3)
    cv2.imshow("Imagem - HP4", imgHP4)
    cv2.imshow("Imagem - FHP", imgFHP)

    # cv2.imwrite("results//Log.jpg", imgHP)

    cv2.waitKey()

if __name__ ==  "__main__":
    main()