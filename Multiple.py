import cv2
import numpy as np
import score 


def main():
    
    img2 = cv2.imread("images//all_souls_000005.jpg", 0)
    divisor = score.ImageDivider()
    img_dividida = divisor.Dividir(img2, 2)
    imgBrisk, numberFeatures = divisor.MultipleBrisk(img_dividida)
    print(numberFeatures)
    # divisor.ShowImages(imgBrisk)
    # densityMean, diversityStd = divisor.FeatureDensity(img_dividida[0].shape[0], img_dividida[0].shape[1], numberFeatures)
    featureStd = divisor.FeatureStatistic(numberFeatures)
    # print(densityMean, diversityStd)
    print(featureStd)

    img = cv2.imread("images//pedras-1.jpg", 0)
    # divisor = score.ImageDivider()
    img_dividida_ref = divisor.Dividir(img, 2)
    imgBrisk_ref, numberFeatures_ref = divisor.MultipleBrisk(img_dividida_ref)
    print(numberFeatures_ref)
    # divisor.ShowImages(imgBrisk)
    # densityMean_ref, diversityStd_ref = divisor.FeatureDensity(img_dividida_ref[0].shape[0], img_dividida_ref[0].shape[1], numberFeatures_ref)
    featureStd_ref = divisor.FeatureStatistic(numberFeatures_ref)
    # print(densityMean_ref, diversityStd_ref)
    print(featureStd_ref)

    # divisor.DensityScore(densityMean_ref, densityMean)
    divisor.Score(featureStd_ref, featureStd, sum(numberFeatures_ref), sum(numberFeatures), 
    divisor.RMSContrast(img), divisor.RMSContrast(img2))

    


if __name__ == "__main__":
    main()
    cv2.waitKey()