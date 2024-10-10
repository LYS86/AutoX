package com.stardust.autojs.runtime.api

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.RectF
import android.util.Log
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.common.ops.CastOp
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.BufferedReader
import java.io.IOException
import java.io.InputStreamReader

class YoloDetector(
    private val context: Context
) {
    private var modelPath: String = "best_float32.tflite" //"yolov8n_float32.tflite"
    private var labelPath: String = ""
    private var inputSize: Int = 640 // 默认输入尺寸
    private var numThreads: Int = 4 // 默认线程数

    private var numChannel = 0
    private var numElements = 0

    private var interpreter: Interpreter? = null
    private var labels = mutableListOf<String>()

    // 创建一个包含所有操作的 ImageProcessor。
    private val imageProcessor = ImageProcessor.Builder()
        .add(ResizeOp(inputSize, inputSize, ResizeOp.ResizeMethod.BILINEAR)) //图片缩放
        .add(NormalizeOp(0f, 255f)) // 归一化操作
        .add(CastOp(DataType.FLOAT32)) // 转换数据类型
        .build()

    // 设置模型路径
    fun setModel(path: String): YoloDetector {
        this.modelPath = path
        return this
    }

    // 设置标签路径
    fun setLabel(path: String): YoloDetector {
        this.labelPath = path
        return this
    }

    // 设置输入尺寸
    fun setInputSize(inputSize: Int): YoloDetector {
        this.inputSize = inputSize
        return this
    }

    // 设置线程数
    fun setNumThreads(numThreads: Int): YoloDetector {
        this.numThreads = numThreads
        return this
    }

    fun init(): YoloDetector {

        if (modelPath.isEmpty()) {
            throw IllegalArgumentException("Model path must be set before initialization")
        }
        initializeInterpreter()
        return this

//        if (labelPath.isEmpty()) {
//            throw IllegalArgumentException("labels path must be set before initialization")
//
//        }
//        loadLabels()
    }

    //加载模型
    private fun initializeInterpreter() {
        val options = Interpreter.Options().apply {
            setNumThreads(numThreads)
            // 如果需要，可以添加GPU或NNAPI委托
        }
        try {
            interpreter = Interpreter(FileUtil.loadMappedFile(context, modelPath), options)
        } catch (e: Exception) {
            Log.e(TAG, "Error initializing interpreter", e)
        }
    }

    private fun loadLabels() {
        try {
            val inputStream = context.assets.open(labelPath)
            val labelList = BufferedReader(InputStreamReader(inputStream)).readLines()
            labels = mutableListOf()
            labelList.forEach { labels.add(it) }
        } catch (e: IOException) {
            Log.e(TAG, "Error loading labels", e)
        }
    }

    fun classify3(bitmap: Bitmap): List<Result> {
        val tensorImage = TensorImage.fromBitmap(bitmap)
        val processedImage = imageProcessor.process(tensorImage)
        val outputTensor = interpreter?.getOutputTensor(0) // 获取第一个输出张量
        if (outputTensor == null) {
            Log.e(TAG, "Output tensor is null")
            return emptyList()
        }
        val outputShape = outputTensor.shape() // 输出张量的形状
        if (outputShape == null) {
            Log.e(TAG, "Output shape is null")
            return emptyList()
        }
        Log.d(TAG, "outputShape: ${outputShape.contentToString()}")
        numChannel = outputShape[1]
        numElements = outputShape[2]

        // 创建输出张量缓冲区
        val outputBuffer = TensorBuffer.createFixedSize(outputShape, outputTensor.dataType())

        // 运行模型推理
        interpreter?.run(processedImage.tensorBuffer.buffer, outputBuffer.buffer)

        Log.d(TAG, "classify3: 检查")
        // 输出和检查 outputBuffer
        Log.d(TAG, "Output buffer size: ${outputBuffer.floatArray.size}")
        Log.d(TAG, "Output buffer contents: ${outputBuffer.floatArray.contentToString()}")

        // 解析输出
        return parseOutput(outputBuffer)
    }


    fun classify1(bitmap: Bitmap): Array<Array<FloatArray>> {
        // 预处理图像
        val tensorImage = TensorImage.fromBitmap(bitmap)
        val processedImage = imageProcessor.process(tensorImage)

        // 获取输出张量信息
        val outputTensor = interpreter?.getOutputTensor(0) // 获取第一个输出张量
        if (outputTensor == null) {
            Log.e(TAG, "Output tensor is null")
            return emptyArray()
        }
        val outputShape = outputTensor.shape() // 输出张量的形状
        if (outputShape == null) {
            Log.e(TAG, "Output shape is null")
            return emptyArray()
        }
        val outputDataType = outputTensor.dataType() // 输出张量的数据类型

        // 打印输出张量的形状和数据类型
        Log.d(TAG, "Output Tensor Shape: ${outputShape.contentToString()}")
        Log.d(TAG, "Output Tensor DataType: $outputDataType")

        // 根据输出张量形状创建相应的缓冲区
        val outputBuffer = Array(1) {
            Array(outputShape[1]) {
                FloatArray(outputShape[2])
            }
        }

        // 运行模型推理
        interpreter?.run(processedImage.tensorBuffer.buffer, outputBuffer)

        // 打印部分输出值以调试模型输出
        outputBuffer[0].take(5).forEachIndexed { index, result ->
            Log.d(TAG, "Output[$index]: ${result.contentToString()}")
        }

        return outputBuffer
    }

    fun classify2(bitmap: Bitmap): List<Box> {
        // 预处理图像
        val tensorImage = TensorImage.fromBitmap(bitmap)
        val processedImage = imageProcessor.process(tensorImage)

        // 获取输出张量信息
        val outputTensor = interpreter?.getOutputTensor(0) // 获取第一个输出张量
        if (outputTensor == null) {
            Log.e(TAG, "Output tensor is null")
            return emptyList()
        }
        val outputShape = outputTensor.shape() // 输出张量的形状
        if (outputShape == null) {
            Log.e(TAG, "Output shape is null")
            return emptyList()
        }
        numChannel = outputShape[1]
        numElements = outputShape[2]
        // 创建输出张量缓冲区
        val outputBuffer = TensorBuffer.createFixedSize(outputShape, DataType.FLOAT32)

        // 运行模型推理
        interpreter?.run(processedImage.tensorBuffer.buffer, outputBuffer.buffer)

        Log.d(TAG, "classify2: ${outputBuffer.floatArray.contentToString()}")
        // 解析输出
        val results = mutableListOf<Box>()
        for (i in 0 until outputShape[1]) {
            val score = outputBuffer.getFloatArray()[4 * i + 4] // 置信度
            if (score > CONFIDENCE_THRESHOLD) { // 只处理高于置信度阈值的结果
//                val className = labels[(outputBuffer.getFloatArray()[4 * i + 5] * labels.size).toInt()] // 类别
                val className =
                    (outputBuffer.getFloatArray()[4 * i + 5] * labels.size).toInt() // 类别

                val xCenter = outputBuffer.getFloatArray()[4 * i] * 2 - 1
                val yCenter = outputBuffer.getFloatArray()[4 * i + 1] * 2 - 1
                val width = outputBuffer.getFloatArray()[4 * i + 2] * 2
                val height = outputBuffer.getFloatArray()[4 * i + 3] * 2
                val topLeftX = (xCenter - width / 2) * bitmap.width
                val topLeftY = (yCenter - height / 2) * bitmap.height

                results.add(
                    Box(
                        `class` = className.toString(), confidence = score, rect = RectF(
                            topLeftX,
                            topLeftY,
                            topLeftX + (width * bitmap.width),
                            topLeftY + (height * bitmap.height)
                        )
                    )
                )
            }
        }

        // 应用非极大值抑制（NMS）以去除重叠的检测框
        val finalResults = applyNMS(results)

        return finalResults
    }

    private fun parseOutput(outputBuffer: TensorBuffer): List<Result> {
        val results = mutableListOf<Result>()
        val outputData = outputBuffer.floatArray
        return results

    }

    private fun applyNMS(boxes: List<Box>): List<Box> {
        val sortedBoxes = boxes.sortedByDescending { it.confidence }.toMutableList()
        val selectedBoxes = mutableListOf<Box>()
        while (sortedBoxes.isNotEmpty()) {
            val first = sortedBoxes.first()
            selectedBoxes.add(first)
            sortedBoxes.remove(first)
            val iterator = sortedBoxes.iterator()
            while (iterator.hasNext()) {
                val nextBox = iterator.next()
                val iou = calculateIoU(first.rect, nextBox.rect)
                if (iou >= IOU_THRESHOLD) {
                    iterator.remove()
                }
            }
        }
        return selectedBoxes
    }

    private fun calculateIoU(box1: RectF, box2: RectF): Float {
        val x1 = maxOf(box1.left, box2.left)
        val y1 = maxOf(box1.top, box2.top)
        val x2 = minOf(box1.right, box2.right)
        val y2 = minOf(box1.bottom, box2.bottom)
        val intersectionArea = maxOf(0F, x2 - x1) * maxOf(0F, y2 - y1)
        val box1Area = box1.width() * box1.height()
        val box2Area = box2.width() * box2.height()
        return intersectionArea / (box1Area + box2Area - intersectionArea)
    }

    fun test1(): Array<Array<FloatArray>> {
        // 从assets文件夹加载测试图片
        val inputStream = context.assets.open("bus.jpg")
        val bitmap = BitmapFactory.decodeStream(inputStream)
        inputStream.close()

        // 进行推理

        val results = init().classify1(bitmap)
        return results

    }

    fun test2(): List<Box> {
        // 从assets文件夹加载测试图片
        val inputStream = context.assets.open("bus.jpg")
        val bitmap = BitmapFactory.decodeStream(inputStream)
        inputStream.close()

        // 进行推理

        val results = init().classify2(bitmap)
        return results

    }

    fun test3(): List<Result> {
        // 从assets文件夹加载测试图片
        val inputStream = context.assets.open("bus.jpg")
        val bitmap = BitmapFactory.decodeStream(inputStream)
        inputStream.close()
        val results = init().classify3(bitmap)
        return results

    }


    data class Result(val bbox: RectF, val classId: Int, val confidence: Float)
    data class Box(
        val `class`: String, val confidence: Float, val rect: RectF
    )


    companion object {
        private const val CONFIDENCE_THRESHOLD = 0.3F
        private const val IOU_THRESHOLD = 0.5F
        private const val TAG = "YOLOClassifierHelper"
    }
}