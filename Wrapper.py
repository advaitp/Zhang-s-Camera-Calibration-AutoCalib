import cv2
import numpy as np
import os
import glob
from scipy.optimize import minimize
from scipy.optimize import least_squares
import argparse

def getV(H, i, j) :
    t1 = H[0][i]*H[0][j]
    t2 = H[0][i]*H[1][j]+H[1][i]*H[0][j]
    t3 = H[1][i]*H[1][j]
    t4 = H[2][i]*H[0][j]+H[0][i]*H[2][j]
    t5 = H[2][i]*H[1][j]+H[1][i]*H[2][j]
    t6 = H[2][i]*H[2][j]
    v = np.array([[t1], [t2], [t3], [t4], [t5], [t6]])
    return v 

def getIntrinsic(homographies) :
    Vtotal = np.zeros((2*len(homographies), 6))
    for i, homog in enumerate(homographies) :
        v11 = getV(homog, 0, 0)
        v12 = getV(homog, 0, 1)
        v22 = getV(homog, 1, 1)
        Vtotal[2*i, :] = v12.T
        Vtotal[2*i+1, :] = (v11-v22).T

    Ui, Si, Vi = np.linalg.svd(Vtotal)
    b = (Vi[-1,:])
    B11, B12, B22, B13, B23, B33 = b[0], b[1], b[2], b[3], b[4], b[5]

    v0 = (B12*B13 - B11*B23)/(B11*B22 - B12**2)
    lambd = B33 - ((B13**2 + v0*(B12*B13 - B11*B23))/B11)
    alpha = np.sqrt(lambd/B11)
    beta = np.sqrt(lambd*B11/(B11*B22-B12**2))
    gamma = -B12*alpha**2/lambd
    u0 = (gamma*v0/beta)-((B13*alpha**2)/(lambd))

    K = np.array([[alpha, gamma, u0], [0, beta, v0], [0, 0, 1]])

    return K, alpha, beta, gamma, lambd, u0, v0
    
def getExtrinsic(A, homographies) : 
    extrinsic = np.zeros((len(homographies), 3, 4))
    for i, H in enumerate(homographies) :
        h1, h2, h3 = H[:, 0], H[:, 1], H[:, 2]
        lamb = np.linalg.norm(np.linalg.inv(A)@h1, ord=2)
        r1 = (1/lamb)*np.linalg.inv(A)@h1
        r2 = (1/lamb)*np.linalg.inv(A)@h2
        r3 = np.cross(r1, r2)
        t = (1/lamb)*np.linalg.inv(A)@h3
        R = np.array([r1, r2, r3]).T
        extrinsic[i, :, :-1] = R
        extrinsic[i, :, -1] = t

    return extrinsic

def getExtrinsicone(A, H) : 
    extrinsic = np.zeros((3, 4))
    h1, h2, h3 = H[:, 0], H[:, 1], H[:, 2]
    lamb = np.linalg.norm(np.linalg.inv(A)@h1, ord=2)
    r1 = (1/lamb)*np.linalg.inv(A)@h1
    r2 = (1/lamb)*np.linalg.inv(A)@h2
    r3 = np.cross(r1, r2)
    t = (1/lamb)*np.linalg.inv(A)@h3
    R = np.array([r1, r2, r3]).T
    extrinsic[:, :-1] = R
    extrinsic[:, -1] = t

    return extrinsic

def to_list(kc,intrinsic_matrix):
    k = np.zeros(6+2)
    intrinsic_matrix_mod = np.array([intrinsic_matrix[0,0],intrinsic_matrix[0,1],intrinsic_matrix[0,2],intrinsic_matrix[1,1],intrinsic_matrix[1,2],intrinsic_matrix[2,2]])

    k[0] = kc[0]
    k[1] = kc[1]
    k[2:] = intrinsic_matrix_mod
    return k

def from_list(k):
    kc = np.zeros(2)
    kc = k[:2]
    intrinsic_matrix = np.zeros((3,3))
    intrinsic_matrix[0,0] = k[2]
    intrinsic_matrix[0,1] = k[3]
    intrinsic_matrix[0,2] = k[4]
    intrinsic_matrix[1,1] = k[5]
    intrinsic_matrix[1,2] = k[6]
    intrinsic_matrix[2,2] = k[7]

    return kc,intrinsic_matrix

def reprojectionError(init, E, homographies, imgpoints, worldpoints, images, output_path, vis=0) :
    totalerror = []
    kc, K = from_list(init)
    for j, H in enumerate(homographies) :
        Ei = E[j]
        imagepts = imgpoints[j]
        img = images[j]
        error = []

        neworldpoints = np.zeros((worldpoints.shape))
        i = 0 
        for wor in worldpoints :
            x, y = wor[0], wor[1]
            wpt = np.array([[x], [y], [0], [1]])
            newpt = np.dot(Ei, wpt)
            newpt = (newpt/newpt[2])
            neworldpoints[i] = [newpt[0], newpt[1]]
            i += 1

        for im, wor in zip(imagepts, neworldpoints) :
            x, y = wor[0], wor[1]
            x_ = x + x*(kc[0]*(x**2+y**2) + kc[1]*(x**2+y**2))
            y_ = y + y*(kc[0]*(x**2+y**2) + kc[1]*(x**2+y**2))

            wpt = np.array([[x_], [y_], [1]])
            impt = np.array([[im[0]], [im[1]], [1]])
            projectedpoint = np.dot(K, wpt)
            projectedpoint = (projectedpoint/projectedpoint[2]).astype(int).flatten()
            if vis : 
                cv2.circle(img, (projectedpoint[0], projectedpoint[1]), 9, (0, 255, 0), -1)
            error.append(np.square((np.linalg.norm(impt.flatten()[:2]-projectedpoint[:2]))))

        merror =np.mean(error)
        # print(f'Mean Error : {merror}')
        totalerror.append(merror)   
        if vis :
            resized = cv2.resize(img, (760,1300), interpolation = cv2.INTER_AREA)
            cv2.imwrite(f'{output_path}/CalibratedPoints_Img{j}.jpg', resized)
 
    print(f'Total Error : {np.sum(totalerror)/13}')
    return np.array(totalerror)

def calibrate(img_path, output_path) :
    cboard = (9, 6)
    # 2D Points
    imgpoints = [] 
    # homographies
    homog = []
    # list of images 
    images = []

    # 3D Points
    worldpoints = np.zeros((cboard[0] * cboard[1], 2), np.float32)
    worldpoints[:,:2] = np.mgrid[1:cboard[0]+1, 1:cboard[1]+1].T.reshape(-1, 2)*21.5
    wpts = np.float32([[1, 1], [9, 1], [9, 6], [1, 6]])*21.5
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    imagesp = glob.glob(f'{img_path}/*.jpg')
    for fname in imagesp:
        img = cv2.imread(fname)
        images.append(img)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, cboard, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

        if ret:
            # Refine pixel coordinates
            corners2 = cv2.cornerSubPix(gray, corners, (11,11),(-1,-1), criteria)
            corners2 = corners2.reshape((-1, 2))
            cornersi = corners2.astype(int)
            impts = np.zeros((4, 2), np.float32)

            # Corner points of all chess points
            impts[0] = cornersi[0]
            impts[1] = cornersi[8]
            impts[2] = cornersi[53]
            impts[3] = cornersi[45]

            H, _ = cv2.findHomography(wpts, impts)
            homog.append(H)
            imgpoints.append(cornersi)

    K, alpha, beta, gamma, lambd, u0, v0 = getIntrinsic(homog)
    print('---------Intital K matrix--------')
    print(K)

    E = getExtrinsic(K, homog)
    kc = np.zeros(2)
    initialization = to_list(kc, K)
    totalerror = reprojectionError(initialization, E, homog, imgpoints, worldpoints, images, output_path)

    ### Optimize the error
    kc = np.zeros(2)
    initialization = to_list(kc, K)
    optimized_params = least_squares(fun = reprojectionError, x0 = initialization, method = "lm", args=[E, homog, imgpoints, worldpoints, images, output_path])
    values = optimized_params.x
    kc, Kn = from_list(values)
    Kn = Kn/Kn[2, 2]

    print('---------K matrix after optimization--------')
    print(Kn)

    print('---------distortion coefficients after optimization--------')
    print(kc)


    ### Visualize the results 
    vis = 1
    totalerror = reprojectionError(values, E, homog, imgpoints, worldpoints, images, output_path, vis)

    ### Warp the image
    # img = cv2.imread('test.jpg')
    # for i in range(len(images)) :
    #     maxWidth, maxHeight = 3000, 3000 
    #     Ei = E[i]
    #     Ein = np.vstack((Ei[:, 0], Ei[:, 1], Ei[:, 3]))
    #     H = np.dot(Kn, Ein)
    #     out = cv2.warpPerspective(img, H, (maxWidth, maxHeight), flags=cv2.INTER_LINEAR)
    #     cv2.imwrite(f'Calibrated_Img{i}.jpg', out)

if __name__ == "__main__" :
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--ImagesPath', default="./Calibration_Imgs", help='base path where image files exist')
    Parser.add_argument('--OutputPath', default="./Output", help='output path where image will be stored')
    Args = Parser.parse_args("")
    images_path = Args.ImagesPath
    output_path = Args.OutputPath
   
    img_path = ''
    calibrate(images_path, output_path)