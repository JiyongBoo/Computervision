## Computer vision programming assignment 2 
## 2019038359 Jiyong BOO

# 1. Mean Filtering
import cv2
import numpy as np

# 원본 이미지 불러오기
img = cv2.imread("car.jpg") 

# float type의 픽셀 값을 uint8(0~255)으로 변환해주는 함수
def float2uint8(img):
    min = img.min()
    max = img.max()
    img = ((img - min) / (max - min) * 255.).astype(np.uint8) 
    return img

# Gaussian noise를 만드는 함수
def Gaussian_Noise(mean,sigma,img):
    height, width, channel = img.shape
    # noised_img = np.zeros((height,width,channel))
    # for i in range(height):
    #     for j in range(width):
    #         noise = np.random.normal(mean, sigma)
    #         for k in range(channel):
    #             noised_img[i,j,k] = img[i,j,k] + noise
    noise = np.random.normal(mean,sigma,(height,width,channel))
    noised_img = img + noise
    return float2uint8(noised_img), float2uint8(noise)

# 평균 = 0, 분산 = 10인 Gausian noise를 원본 이미지에 적용
noised_img , gaussian_noise = Gaussian_Noise(0,10,img)

# padding 함수 : 필터링으로 이미지 크기가 작아지는것을 막기위함
def padding(img, kernel_size):
    pad_size = kernel_size//2
    height, width, channel = img.shape
    padding_img = np.zeros((height + (2*pad_size), width+(2*pad_size), channel), dtype=np.float64)    
    if pad_size == 0:
        padding_img = img.copy()
    else:
        padding_img[pad_size:-pad_size, pad_size:-pad_size] = img.copy()
    return float2uint8(padding_img)

# mean filter를 적용하는 함수
def mean_filter(kernel_size, img):
    padding_img = padding(img,kernel_size)
    height, width, channel = img.shape
    filtered_img = np.zeros([height,width,channel],dtype=float)
    for i in range(height):
        for j in range(width):
            for k in range(channel):
                kernel = padding_img[i:i+kernel_size,j:j+kernel_size,k]
                filtered_img[i,j,k] = np.mean(kernel)          
    return float2uint8(filtered_img)

# kernel size 3,5,7인 median filter를 noised image에 적용
kernel_size = [3, 5, 7]

for i in kernel_size:
    globals()["filtered_img_{}".format(i)] = mean_filter(i,noised_img)
    

# 1번 문항 결과 시각화
import matplotlib.pyplot as plt

# 결과 비교
plt.figure(figsize=(15,8))
plt.subplot(2,3,1)
plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
plt.title('Original image')
plt.subplot(2,3,2)
plt.imshow(cv2.cvtColor(noised_img,cv2.COLOR_BGR2RGB))
plt.title('Gaussian noised image')
plt.subplot(2,3,3)
plt.imshow(cv2.cvtColor(gaussian_noise,cv2.COLOR_BGR2RGB))
plt.title('Gaussian noise')
plt.subplot(2,3,4)
plt.imshow(cv2.cvtColor(filtered_img_3,cv2.COLOR_BGR2RGB))
plt.title('Mean filtering image(kernel size=3)')
plt.subplot(2,3,5)
plt.imshow(cv2.cvtColor(filtered_img_5,cv2.COLOR_BGR2RGB))
plt.title('Mean filtering image(kernel size=5)')
plt.subplot(2,3,6)
plt.imshow(cv2.cvtColor(filtered_img_7,cv2.COLOR_BGR2RGB))
plt.title('Mean filtering image(kernel size=7)')
plt.tight_layout()  
plt.show()

# 1번 문항 결과 원본 파일 시각화
# cv2.imshow('Original image',img)
# cv2.imshow('Gaussian noised image',noised_img)
# cv2.imshow('Mean filtering image(kernel size=3)',filtered_img_3)
# cv2.imshow('Mean filtering image(kernel size=5)',filtered_img_5)
# cv2.imshow('Mean filtering image(kernel size=7)',filtered_img_7)
# cv2.waitKey(0)

# PSNR 계산 함수 (이미지 사이즈 같은 경우만 가능)
def PSNR(origin_img,compared_img):
    try:
        height, width, channel = origin_img.shape
        MSE = np.average(np.sum(np.square(origin_img-compared_img)))/(height*width*channel)
        min = origin_img.min()
        max = origin_img.max()
        PSNR = 10*np.log10((max-min)**2/MSE)
        return PSNR, MSE
    except:
        print('The two images have differnt size')
        return 0

# kernel size 별로 noised image와 filtering image PSNR 비교
for i in kernel_size:
    globals()["PSNR_{}".format(i)], globals()["MSE_{}".format(i)] = PSNR(noised_img,globals()["filtered_img_{}".format(i)])
    print('PSNR(filtering kernel size = {}) : '.format(i),globals()["PSNR_{}".format(i)])
    print('MSE(filtering kernel size = {}) : '.format(i),globals()["MSE_{}".format(i)])

# 2. Unsharp Masking

# Gaussian filter를 적용하는 함수
def Gaussian_filter(kernel_size, sigma, img):
    padding_img = padding(img, kernel_size)
    height, width, channel = img.shape
    kernel_limit = kernel_size//2
    y, x = np.ogrid[-kernel_limit:kernel_limit+1, -kernel_limit:kernel_limit+1]
    filter = 1/(2*np.pi*(sigma**2))*np.exp(-1*(x**2+y**2)/(2*(sigma**2)))
    filtered_img = np.zeros([height,width,channel],dtype=np.float64)
    for i in range(height):
        for j in range(width):
            for k in range(channel):
                kernel = padding_img[i:i+kernel_size,j:j+kernel_size,k]
                filtered_img[i,j,k] = np.sum(kernel*filter)            
    return float2uint8(filtered_img)   


# Unsharp masking 함수
def Unsharp_masking(img,kernel_size,sigma=1,alpha:float=1.0):    
    blur_img = Gaussian_filter(kernel_size,sigma,img)
    unsharp_img = np.clip((1+alpha)*img - alpha*blur_img, 0, 255)
    return float2uint8(unsharp_img)

# kernel size 3,5,7인 Unsharp masking 적용
for i in kernel_size:
    globals()["unsharp_img_{}".format(i)] = Unsharp_masking(img,i,1,1.)

# 결과 비교
plt.figure(figsize=(8,8))
plt.subplot(2,2,1)
plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
plt.title('Original image')
plt.subplot(2,2,2)
plt.imshow(cv2.cvtColor(unsharp_img_3,cv2.COLOR_BGR2RGB))
plt.title('Unsharp masking image(kernel size=3)')
plt.subplot(2,2,3)
plt.imshow(cv2.cvtColor(unsharp_img_5,cv2.COLOR_BGR2RGB))
plt.title('Unsharp masking image(kernel size=5)')
plt.subplot(2,2,4)
plt.imshow(cv2.cvtColor(unsharp_img_7,cv2.COLOR_BGR2RGB))
plt.title('Unsharp masking image(kernel size=7)')
plt.tight_layout()  
plt.show()

# 2번 문항 결과 원본 파일 시각화
# cv2.imshow('unsharp_img_3',unsharp_img_3)
# cv2.imshow('unsharp_img_5',unsharp_img_5)
# cv2.imshow('unsharp_img_7',unsharp_img_7)
# cv2.waitKey(0)

# 3. Contrast Stretching

# Contrast stretching 함수
def Contrast_stretcing(img,a,b,va,vb):
    max = img.max()
    alpha = va/a
    beta = (vb-va)/(b-a)
    gamma = (max-vb)/(max-b)
    height, width, channel = img.shape
    constr_img = np.zeros((height,width,channel))
    for i in range(height):
        for j in range(width):
            for k in range(channel):
                if img[i,j,k] < a:
                    constr_img[i,j,k] = alpha*img[i,j,k]
                elif a <= img[i,j,k] < b :
                    constr_img[i,j,k] = beta*(img[i,j,k]-a)+va
                else:
                    constr_img[i,j,k] = gamma*(img[i,j,k]-b)+vb
    return float2uint8(constr_img)

# a=80, va=50, b=180, vb=210 으로 contrast stretching 적용
contr_img = Contrast_stretcing(img,80,180,50,210)
# a=80, b=180으로 clipping 적용
clipping_img = Contrast_stretcing(img,80,180,0,255)

# Gamma correction 함수
def Gamma_correction(img,gamma):
    gmc_img = img**gamma
    return float2uint8(gmc_img)

# gamma = 0.5,2.0 으로 gamma correction 적용
gmc_img_low = Gamma_correction(img, 0.5)
gmc_img_high = Gamma_correction(img, 2.0)

# 결과 비교
plt.figure(figsize=(8,8))
plt.subplot(2,2,1)
plt.imshow(cv2.cvtColor(contr_img,cv2.COLOR_BGR2RGB))
plt.title('Contrast stretching image')
plt.subplot(2,2,2)
plt.imshow(cv2.cvtColor(clipping_img,cv2.COLOR_BGR2RGB))
plt.title('Clipping image')
plt.subplot(2,2,3)
plt.imshow(cv2.cvtColor(gmc_img_low,cv2.COLOR_BGR2RGB))
plt.title('Gamma correction image(gamma = 0.5)')
plt.subplot(2,2,4)
plt.imshow(cv2.cvtColor(gmc_img_high,cv2.COLOR_BGR2RGB))
plt.title('Gamma correction image(gamma = 2.0)')
plt.tight_layout()  
plt.show()

# 3번 문항 결과 원본 이미지 시각화
# cv2.imshow('contrast_stretching_img',contr_img)
# cv2.imshow('clipping_img',clipping_img)
# cv2.imshow('gamma=0.5',gmc_img_low)
# cv2.imshow('gamma=2.0',gmc_img_high)
# cv2.waitKey(0)

# 4. Histogram Equalization

# histogram equalization 함수
def Histogram_Equalization(img):
    height, width, channel = img.shape
    max = 256
    HE_img = np.zeros([height,width,channel])
    for c in range(channel):
        Histogram = np.zeros(max)
        for i in img[:,:,c].ravel():
            Histogram[i] += 1
        for i in range(1,max):
            Histogram[i] += Histogram[i-1]
        Histogram = np.round(Histogram/(height*width)*(max-1))
        for h in range(height):
            for w in range(width):
                HE_img[h,w,c] = Histogram[img[h,w,c]]
    return float2uint8(HE_img)

HE_img = Histogram_Equalization(img)

# 원본 이미지와 Histogram equalization 적용 이미지 비교
plt.figure(figsize=(6,12))
plt.subplot(1,2,1)
plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
plt.title('Original image')
plt.subplot(1,2,2)
plt.imshow(cv2.cvtColor(HE_img,cv2.COLOR_BGR2RGB))
plt.title('Histogram equalization image')
plt.tight_layout()  
plt.show()

# 4번 문항 결과 원본 이미지 시각화
# cv2.imshow('Histogram_Equalization_img',HE_img)
# cv2.waitKey(0)

# 이미지 히스토그램 시각화 함수 : 파란색 막대 = histogram / 빨간색 그래프 = 누적 histogram
def img2Hist(img):
    _, _, channel = img.shape
    title = ['Histogram Blue channel','Histogram Green channel','Histogram Red channel']
    plt.figure(figsize = (20,5))
    for c in range(channel):
        pix_val = np.ravel(img[:,:,c])
        plt.subplot(1,3,c+1)
        plt.hist(pix_val, range = [0,255], alpha = 0.7, bins = 256)
        hist2 = plt.twinx()
        hist2.hist(pix_val, bins = 256, range = [0,255], color = 'r',cumulative=True, histtype='step')
        plt.title(title[c])

# 원본 이미지와 HE 적용 이미지 히스토그램 시각화   
img2Hist(img)
plt.tight_layout()  
plt.show()
img2Hist(HE_img)   
plt.tight_layout()  
plt.show()

# 5. Image Upsampling
# 1/4배 downsample 적용 함수 : Mean pooling
def downsample(img):
    height, width, channel = img.shape
    downsample_img = np.zeros([height//4,width//4,channel])
    for i in range(height//4):
        for j in range(width//4):
            for k in range(channel):
                kernel = img[4*i:4*i+4,4*j:4*j+4,k]
                downsample_img[i,j,k] = np.mean(kernel)
    return float2uint8(downsample_img)

downsample_img = downsample(img)

# upsampling 함수 직접 구현
# # nearest neighbor interporation upsample 함수
# def nearest_neighbor(img):
#     height, width, channel = img.shape
#     upsample_img = np.zeros([height*4,width*4,channel])
#     for i in range(height):
#         for j in range(width):
#             for k in range(channel):
#                 kernel = np.full([4,4],img[i,j,k])
#                 upsample_img[4*i:4*i+4,4*j:4*j+4,k] = kernel 
#     return float2uint8(upsample_img)

# nearest_neighbor_img = nearest_neighbor(downsample_img)

# # bilinear interporation upsample 함수
# def bilinear(img):
#     height, width, channel = img.shape
#     upsample_img = np.zeros([height*4,width*4,channel])
#     for i in range(4*height):
#         for j in range(4*width):
#             for k in range(channel):
#                 p, q, p_prime, q_prime = i//4,j//4,i/4,j/4
#                 a = p_prime - p
#                 b = q_prime - q
#                 if p == height-1:
#                     p,a = p-1, a+1
#                 if q == width-1:
#                     q,b = q-1, b+1  
#                 upsample_img[i,j,k] = (1-a)*((1-b)*img[p,q,k]+b*img[p,q+1,k])\
#                     +a*((1-b)*img[p+1,q,k]+b*img[p+1,q+1,k])
#     return float2uint8(upsample_img)

# bilinear_img = bilinear(downsample_img)

# # bicubic interporation upsample 함수
# def bicubic(img):

#     def W(x,a):
#         t = abs(x)
#         if t <= 1:
#             return (a+2)*(t**3) - (a+3)*(t**2) + 1
#         elif 1 < t <=2:
#             return  a*(t**3) - 5*a*(t**2) + 8*a*t -4*a
#         else:
#             return 0
        
#     height, width, channel = img.shape
#     upsample_img = np.zeros([height*4,width*4,channel])
#     for i in range(4*height):
#         for j in range(4*width):
#             for k in range(channel):
#                 p, q, p_prime, q_prime = i//4,j//4,i/4,j/4
#                 a = p_prime - p
#                 b = q_prime - q
#                 if p == 0:
#                     p,a = p+1 , a-1
#                 if q == 0:
#                     q,b = q+1, b-1
#                 if p >= height-2:
#                     p = height-3
#                     a = p_prime - p
#                 if q >= width-2:
#                     q = width -3
#                     b = q_prime - q
#                 num = 0
#                 den = 0
#                 for m in range(-1,3):
#                     for n in range(-1,3):
#                         num += img[p+m,q+n,k]*W(m-a,-1)*W(n-b,-1)
#                         den += W(m-a,-1)*W(n-b,-1)
#                 upsample_img[i,j,k] = num/den
#     return float2uint8(upsample_img)

# bicubic_img = bicubic(downsample_img)

# cv2.resize() 함수를 이용한 upsample
original_height, original_width, _ = img.shape
nearest_neighbor_img = cv2.resize(downsample_img, dsize=[original_height,original_width], interpolation=cv2.INTER_NEAREST)
bilinear_img = cv2.resize(downsample_img, dsize=[original_height,original_width], interpolation=cv2.INTER_LINEAR)
bicubic_img = cv2.resize(downsample_img, dsize=[original_height,original_width], interpolation=cv2.INTER_CUBIC)

# 결과 비교
plt.figure(figsize=(12,8))
ax1 = plt.subplot2grid((8, 12), (0, 0), colspan=4, rowspan=4)
ax2 = plt.subplot2grid((8, 12), (0, 4), colspan=1, rowspan=1)
ax3 = plt.subplot2grid((8, 12), (4, 0), colspan=4, rowspan=4)
ax4 = plt.subplot2grid((8, 12), (4, 4), colspan=4, rowspan=4)
ax5 = plt.subplot2grid((8, 12), (4, 8), colspan=4, rowspan=4)

ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
ax1.set_title('Original image')
ax2.imshow(cv2.cvtColor(downsample_img, cv2.COLOR_BGR2RGB))
ax2.set_title('Downsampled image')
ax3.imshow(cv2.cvtColor(nearest_neighbor_img, cv2.COLOR_BGR2RGB))
ax3.set_title('Nearest neighbor interpolation')
ax4.imshow(cv2.cvtColor(bilinear_img, cv2.COLOR_BGR2RGB))
ax4.set_title('Bilinear interpolation')
ax5.imshow(cv2.cvtColor(bicubic_img, cv2.COLOR_BGR2RGB))
ax5.set_title('Bicubic interpolation')

plt.tight_layout()  
plt.show()

# 문항 5 결과 원본 이미지 시각화
# cv2.imshow("downsampled image",downsample_img)
# cv2.imshow("nearest_neighbor image",nearest_neighbor_img)
# cv2.imshow("bilinear image",bilinear_img)
# cv2.imshow("bicubic image",bicubic_img)
# cv2.waitKey(0)

# 각 interporation 이미지 별 PSNR 값 비교 (원본 이미지 height,width가 4의 배수가 아닐시 계산 불가)
print('PSNR,MSE(nearest neighbor) : ', PSNR(img,nearest_neighbor_img))
print('PSNR,MSE(bilinear) : ',PSNR(img,bilinear_img))
print('PSNR,MSE(bicubic) : ',PSNR(img,bicubic_img))