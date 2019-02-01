//
//  ViewController.swift
//  EdgeConnect-CoreML
//
//  Created by 马丹 on 2019/1/27.
//  Copyright © 2019 马丹. All rights reserved.
//

import UIKit
import CoreML
import GPUImage

class ViewController: UIViewController {
    typealias FilteringCompletion = ((UIImage?, Error?) -> ())

    @IBOutlet private var imageView: UIImageView!
    @IBOutlet private var loader: UIActivityIndicatorView!
    @IBOutlet private var buttonHolderView: UIView!
    @IBOutlet private var applyButton: UIButton!
    @IBOutlet private var loaderWidthConstraint: NSLayoutConstraint!

    var picture:RawDataInput!
    var filter:CannyEdgeDetection!
    var edges:RawDataOutput!

    var imagePicker = UIImagePickerController()
    var isProcessing : Bool = false {
        didSet {
            self.applyButton.isEnabled = !isProcessing
            self.isProcessing ? self.loader.startAnimating() : self.loader.stopAnimating()
            self.loaderWidthConstraint.constant = self.isProcessing ? 20.0 : 0.0
            UIView.animate(withDuration: 0.3) {
                self.isProcessing
                    ? self.applyButton.setTitle("Processing...", for: .normal)
                    : self.applyButton.setTitle("Apply EdgeConnect", for: .normal)
                self.view.layoutIfNeeded()
            }
        }
    }

    //MARK:- Lifecycle

    override func viewDidLoad() {
        super.viewDidLoad()
        self.isProcessing = false
        self.applyButton.superview!.layer.cornerRadius = 4
    }

    //MARK: - Utils
    func showError(_ error: Error) {
        self.buttonHolderView.backgroundColor = UIColor(red: 220/255, green: 50/255, blue: 50/255, alpha: 1)
        self.applyButton.setTitle(error.localizedDescription, for: .normal)

        DispatchQueue.main.asyncAfter(deadline: .now() + 2.0) { // Change `2.0` to the desired number of seconds.
            self.applyButton.setTitle("Apply Style", for: .normal)
            self.buttonHolderView.backgroundColor = UIColor(red: 5/255, green: 122/255, blue: 255/255, alpha: 1)
        }
    }

    //MARK:- CoreML

    func testMLMultiArray(pixelBuffer: CVPixelBuffer, data: MLMultiArray, height: Int, width: Int) {
        CVPixelBufferLockBaseAddress(pixelBuffer, CVPixelBufferLockFlags(rawValue: 0))
        let baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer)
        let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
        let buffer = baseAddress!.assumingMemoryBound(to: UInt8.self)

        var ptrData = UnsafeMutablePointer<Float>(OpaquePointer(data.dataPointer))
        ptrData = ptrData.advanced(by: 0)

        let cStride = width * height

        for y in 0..<height {
            for x in 0..<width {
                ptrData[y*width + x + cStride * 0] = (Float)(buffer[y*bytesPerRow+x*4+1])
                ptrData[y*width + x + cStride * 1] = (Float)(buffer[y*bytesPerRow+x*4+2])
                ptrData[y*width + x + cStride * 2] = (Float)(buffer[y*bytesPerRow+x*4+3])
            }
        }
    }

    func getGrayImage(pixelBuffer: CVPixelBuffer, data: MLMultiArray, height: Int, width: Int) {
        CVPixelBufferLockBaseAddress(pixelBuffer, CVPixelBufferLockFlags(rawValue: 0))
        let baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer)
        let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
        let buffer = baseAddress!.assumingMemoryBound(to: UInt8.self)

        var ptrData = UnsafeMutablePointer<Float>(OpaquePointer(data.dataPointer))
        ptrData = ptrData.advanced(by: 0)

        for y in 0..<height {
            for x in 0..<width {
                let index = y*bytesPerRow+x*4
                ptrData[y*width + x] = ((Float)(buffer[index+1]) * 0.299 + (Float)(buffer[index+2]) * 0.587 + (Float)(buffer[index+3]) * 0.114) / 255.0
            }
        }
    }

    func calcNorm(r: Float, g: Float, b: Float) -> Float {
        return ((r-255.0)*(r-255.0) + (g-255.0)*(g-255.0) + (b-255.0)*(b-255.0)).squareRoot()
    }

    func getMask(pixelBuffer: CVPixelBuffer, data: MLMultiArray, height: Int, width: Int) {
        CVPixelBufferLockBaseAddress(pixelBuffer, CVPixelBufferLockFlags(rawValue: 0))
        let baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer)
        let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
        let buffer = baseAddress!.assumingMemoryBound(to: UInt8.self)

        var ptrData = UnsafeMutablePointer<Float>(OpaquePointer(data.dataPointer))
        ptrData = ptrData.advanced(by: 0)

        for y in 0..<height {
            for x in 0..<width {
                let index = y*bytesPerRow+x*4
                if calcNorm(r: (Float)(buffer[index+1]), g: (Float)(buffer[index+2]), b: (Float)(buffer[index+3])) > 8.66 {
                    ptrData[y*width + x] = 0
                } else {
                    ptrData[y*width + x] = 1
                }
            }
        }
    }

    func prepareEdgeInput(gray: MLMultiArray, edge: MLMultiArray, mask: MLMultiArray, input: MLMultiArray, height: Int, width: Int) {
        var ptrGray = UnsafeMutablePointer<Float>(OpaquePointer(gray.dataPointer))
        ptrGray = ptrGray.advanced(by: 0)
        var ptrEdge = UnsafeMutablePointer<Float>(OpaquePointer(edge.dataPointer))
        ptrEdge = ptrEdge.advanced(by: 0)
        var ptrMask = UnsafeMutablePointer<Float>(OpaquePointer(mask.dataPointer))
        ptrMask = ptrMask.advanced(by: 0)
        var ptrInput = UnsafeMutablePointer<Float>(OpaquePointer(input.dataPointer))
        ptrInput = ptrInput.advanced(by: 0)

        let cStride = width * height

        for y in 0..<height {
            for x in 0..<width {
                if ptrMask[y*width + x] != 0 {
                    ptrInput[y*width + x + cStride * 0] = ptrMask[y*width + x]
                    ptrInput[y*width + x + cStride * 1] = 0
                } else {
                    ptrInput[y*width + x + cStride * 0] = ptrGray[y*width + x]
                    ptrInput[y*width + x + cStride * 1] = ptrEdge[y*width + x]
                }
                ptrInput[y*width + x + cStride * 2] = ptrMask[y*width + x]
            }
        }
    }

    func prepareInpaintingInput(image: CVPixelBuffer, mask: MLMultiArray, edge: MLMultiArray, input: MLMultiArray, height: Int, width: Int) {
        CVPixelBufferLockBaseAddress(image, CVPixelBufferLockFlags(rawValue: 0))
        let baseAddress = CVPixelBufferGetBaseAddress(image)
        let bytesPerRow = CVPixelBufferGetBytesPerRow(image)
        let buffer = baseAddress!.assumingMemoryBound(to: UInt8.self)

        var ptrMask = UnsafeMutablePointer<Float>(OpaquePointer(mask.dataPointer))
        ptrMask = ptrMask.advanced(by: 0)
        var ptrEdge = UnsafeMutablePointer<Float>(OpaquePointer(edge.dataPointer))
        ptrEdge = ptrEdge.advanced(by: 0)
        var ptrInput = UnsafeMutablePointer<Float>(OpaquePointer(input.dataPointer))
        ptrInput = ptrInput.advanced(by: 0)

        let cStride = width * height

        for y in 0..<height {
            for x in 0..<width {
                let index = y*bytesPerRow+x*4
                if ptrMask[y*width + x] != 0 {
                    ptrInput[y*width + x + cStride * 0] = ptrMask[y*width + x]
                    ptrInput[y*width + x + cStride * 1] = ptrMask[y*width + x]
                    ptrInput[y*width + x + cStride * 2] = ptrMask[y*width + x]
                } else {
                    ptrInput[y*width + x + cStride * 0] = (Float)(buffer[index+1]) / 255.0
                    ptrInput[y*width + x + cStride * 1] = (Float)(buffer[index+2]) / 255.0
                    ptrInput[y*width + x + cStride * 2] = (Float)(buffer[index+3]) / 255.0
                }
                ptrInput[y*width + x + cStride * 3] = ptrEdge[y*width + x]
            }
        }
    }

    func mergeOutputImage(image: CVPixelBuffer, inpainting: MLMultiArray, mask: MLMultiArray, output: MLMultiArray, height: Int, width: Int) {
        CVPixelBufferLockBaseAddress(image, CVPixelBufferLockFlags(rawValue: 0))
        let baseAddress = CVPixelBufferGetBaseAddress(image)
        let bytesPerRow = CVPixelBufferGetBytesPerRow(image)
        let buffer = baseAddress!.assumingMemoryBound(to: UInt8.self)

        var ptrInpainting = UnsafeMutablePointer<Float>(OpaquePointer(inpainting.dataPointer))
        ptrInpainting = ptrInpainting.advanced(by: 0)
        var ptrMask = UnsafeMutablePointer<Float>(OpaquePointer(mask.dataPointer))
        ptrMask = ptrMask.advanced(by: 0)
        var ptrOutput = UnsafeMutablePointer<Float>(OpaquePointer(output.dataPointer))
        ptrOutput = ptrOutput.advanced(by: 0)

        let cStride = width * height

        for y in 0..<height {
            for x in 0..<width {
                let index = y*bytesPerRow+x*4
                if ptrMask[y*width + x] != 0 {
                    ptrOutput[y*width + x + cStride * 0] = ptrInpainting[y*width + x + cStride * 0]
                    ptrOutput[y*width + x + cStride * 1] = ptrInpainting[y*width + x + cStride * 1]
                    ptrOutput[y*width + x + cStride * 2] = ptrInpainting[y*width + x + cStride * 2]
                } else {
                    ptrOutput[y*width + x + cStride * 0] = (Float)(buffer[index+1]) / 255.0
                    ptrOutput[y*width + x + cStride * 1] = (Float)(buffer[index+2]) / 255.0
                    ptrOutput[y*width + x + cStride * 2] = (Float)(buffer[index+3]) / 255.0
                }
            }
        }
    }

    func writeEdgeInputArray(input: CVPixelBuffer, height: Int, width: Int) -> [UInt8] {
        CVPixelBufferLockBaseAddress(input, CVPixelBufferLockFlags(rawValue: 0))
        let baseAddress = CVPixelBufferGetBaseAddress(input)
        let bytesPerRow = CVPixelBufferGetBytesPerRow(input)
        let buffer = baseAddress!.assumingMemoryBound(to: UInt8.self)
    
        var output = [UInt8](repeating:0, count:height * width * 4)

        for y in 0..<height {
            for x in 0..<width {
                let index = y*bytesPerRow+x*4
                output[index] = buffer[index+1]
                output[index+1] = buffer[index+2]
                output[index+2] = buffer[index+3]
                output[index+3] = 255
            }
        }

        return output
    }

    func readEdgeArray(input: [UInt8], output: MLMultiArray, height: Int, width: Int) {
        var ptrOutput = UnsafeMutablePointer<Float>(OpaquePointer(output.dataPointer))
        ptrOutput = ptrOutput.advanced(by: 0)

        for y in 0..<height {
            for x in 0..<width {
                ptrOutput[y*width+x] = Float(input[y*width+x]) / 255.0
            }
        }
    }

    func process(input: UIImage, completion: @escaping FilteringCompletion) {
        let startTime = CFAbsoluteTimeGetCurrent()

        // Initialize the EdgeConnect model
        let model_edge = edge()
        let model_inpainting = inpainting()

        let height = 320
        let width = 320

        // Next steps are pretty heavy, better process them on another thread
        DispatchQueue.global().async {
            // 1 - Resize our input image
            guard let inputImage = input.resize(to: CGSize(width: width, height: height)) else {
                completion(nil, EdgeConnectError.resizeError)
                return
            }

            // 2 - Edge Model
            guard let cvBufferInput = inputImage.pixelBuffer() else {
                completion(nil, EdgeConnectError.pixelBufferError)
                return
            }

            guard let mlGray = try? MLMultiArray(shape: [1, NSNumber(value: width), NSNumber(value: height)], dataType: MLMultiArrayDataType.float32) else {
                completion(nil, EdgeConnectError.allocError)
                return
            }

            guard let mlEdge = try? MLMultiArray(shape: [1, NSNumber(value: width), NSNumber(value: height)], dataType: MLMultiArrayDataType.float32) else {
                completion(nil, EdgeConnectError.allocError)
                return
            }

            self.getGrayImage(pixelBuffer: cvBufferInput, data: mlGray, height: height, width: width)
            //let image = mlGray.image(min: 0, max: 1, axes: (0, 1, 2))

            let edgeInputImage = self.writeEdgeInputArray(input: cvBufferInput, height: height, width: width)
            var edgeImage = [UInt8](repeating:0, count:height * width)

            self.edges = RawDataOutput()
            self.edges.dataAvailableCallback = { data in
                for y in 0..<height {
                    for x in 0..<width {
                        edgeImage[y*width+x] = data[(y*width+x)*4]
                    }
                }
            }
            self.picture = RawDataInput()
            self.filter = CannyEdgeDetection()
            self.picture --> self.filter --> self.edges
            self.picture.uploadBytes(edgeInputImage, size:Size(width:Float(width), height:Float(height)), pixelFormat:.rgba)

            self.readEdgeArray(input: edgeImage, output: mlEdge, height: height, width: width)
            //let image = mlEdge.image(min: 0, max: 1, axes: (0, 1, 2))

            guard let mlMask = try? MLMultiArray(shape: [1, NSNumber(value: width), NSNumber(value: height)], dataType: MLMultiArrayDataType.float32) else {
                completion(nil, EdgeConnectError.allocError)
                return
            }

            self.getMask(pixelBuffer: cvBufferInput, data: mlMask, height: height, width: width)
            //let image = mlMask.image(min: 0, max: 1, axes: (0, 1, 2))

            guard let mlInputEdge = try? MLMultiArray(shape: [3, NSNumber(value: width), NSNumber(value: height)], dataType: MLMultiArrayDataType.float32) else {
                completion(nil, EdgeConnectError.allocError)
                return
            }

            self.prepareEdgeInput(gray: mlGray, edge: mlEdge, mask: mlMask, input: mlInputEdge, height: height, width: width)

            guard let inputEdge = try? edgeInput(input_1: mlInputEdge) else {
                completion(nil, EdgeConnectError.allocError)
                return
            }

            guard let edgeOutput = try? model_edge.prediction(input: inputEdge) else {
                completion(nil, EdgeConnectError.predictionError)
                return
            }
            //let image = edgeOutput._153.image(min: 0, max: 1, axes: (0, 1, 2))

            // 3 - InPainting model
            guard let mlInputInpainting = try? MLMultiArray(shape: [4, NSNumber(value: width), NSNumber(value: height)], dataType: MLMultiArrayDataType.float32) else {
                completion(nil, EdgeConnectError.allocError)
                return
            }

            self.prepareInpaintingInput(image: cvBufferInput, mask: mlMask, edge: edgeOutput._153, input: mlInputInpainting, height: height, width: width)

            guard let inputInpainting = try? inpaintingInput(input_1: mlInputInpainting) else {
                completion(nil, EdgeConnectError.allocError)
                return
            }

            guard let inpaintingOutput = try? model_inpainting.prediction(input: inputInpainting) else {
                completion(nil, EdgeConnectError.predictionError)
                return
            }
            //let image = inpaintingOutput._173.image(min: 0, max: 1, axes: (0, 1, 2))

            guard let mlOutput = try? MLMultiArray(shape: [3, NSNumber(value: width), NSNumber(value: height)], dataType: MLMultiArrayDataType.float32) else {
                completion(nil, EdgeConnectError.allocError)
                return
            }

            self.mergeOutputImage(image: cvBufferInput, inpainting: inpaintingOutput._173, mask: mlMask, output: mlOutput, height: height, width: width)
            let image = mlOutput.image(min: 0, max: 1, axes: (0, 1, 2))

            // 4 - Hand result to main thread
            DispatchQueue.main.async {
                let timeElapsed = CFAbsoluteTimeGetCurrent() - startTime
                print("Time elapsed for EdgeConnect process: \(timeElapsed) s.")

                completion(image, nil)
            }
        }
    }

    //MARK:- Actions

    @IBAction func importFromLibrary() {
        if UIImagePickerController.isSourceTypeAvailable(.photoLibrary) {
            self.imagePicker.delegate = self
            self.imagePicker.sourceType = .photoLibrary
            self.imagePicker.allowsEditing = false
            self.present(self.imagePicker, animated: true)
        } else {
            print("Photo Library not available")
        }
    }

    @IBAction func applyNST() {
        guard let image = self.imageView.image else {
            print("Select an image first")
            return
        }

        self.isProcessing = true
        self.process(input: image) { filteredImage, error in
            self.isProcessing = false
            if let filteredImage = filteredImage {
                self.imageView.image = filteredImage
            } else if let error = error {
                self.showError(error)
            } else {
                self.showError(EdgeConnectError.unknown)
            }
        }
    }
}

// MARK: - UIImagePickerControllerDelegate

extension ViewController: UIImagePickerControllerDelegate, UINavigationControllerDelegate {
    func imagePickerControllerDidCancel(_ picker: UIImagePickerController) {
        self.dismiss(animated: true)
    }

    func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey : Any]) {
        // Local variable inserted by Swift 4.2 migrator.
        let info = convertFromUIImagePickerControllerInfoKeyDictionary(info)

        if let pickedImage = info[convertFromUIImagePickerControllerInfoKey(UIImagePickerController.InfoKey.originalImage)] as? UIImage {
            self.imageView.image = pickedImage
            self.imageView.backgroundColor = .clear
        }
        self.dismiss(animated: true)
    }
}

// Helper function inserted by Swift 4.2 migrator.
fileprivate func convertFromUIImagePickerControllerInfoKeyDictionary(_ input: [UIImagePickerController.InfoKey: Any]) -> [String: Any] {
    return Dictionary(uniqueKeysWithValues: input.map {key, value in (key.rawValue, value)})
}

// Helper function inserted by Swift 4.2 migrator.
fileprivate func convertFromUIImagePickerControllerInfoKey(_ input: UIImagePickerController.InfoKey) -> String {
    return input.rawValue
}
