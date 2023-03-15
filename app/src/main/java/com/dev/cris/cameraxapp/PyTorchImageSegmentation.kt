package com.dev.cris.cameraxapp

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import org.pytorch.IValue
import org.pytorch.Module
import org.pytorch.Tensor
import org.pytorch.torchvision.TensorImageUtils
import java.io.File
import java.io.FileOutputStream

class PyTorchImageSegmentation() {

    private var module: Module? = null

    fun assetFilePath(context: Context, assetName: String): String {
        val file = File(context.filesDir, assetName)

        if (file.exists() && file.length() > 0) {
            Log.d("path", ""+file.absolutePath)
            try {
                module = Module.load(file.absolutePath)
                Log.d("PyTorchModel", "loading model: ${file.absolutePath}")
            } catch (e: Exception) {
                Log.e("PyTorchModel", "Error loading model: $e")
            }
            return file.absolutePath
        }

        context.assets.open(assetName).use { inputStream ->
            FileOutputStream(file).use { outputStream ->
                val buffer = ByteArray(4 * 1024)
                var read: Int
                while (inputStream.read(buffer).also { read = it } != -1) {
                    outputStream.write(buffer, 0, read)
                }
                outputStream.flush()
            }
        }

        Log.d("path", ""+file.absolutePath)
        try {
            module = Module.load(file.absolutePath)
            Log.d("PyTorchModel", "loading model: ${file.absolutePath}")
        } catch (e: Exception) {
            Log.e("PyTorchModel", "Error loading model: $e")
        }
        return file.absolutePath
    }


    fun inference(bitmap: Bitmap):List<Detection> {
        // Preprocesar la imagen
        Log.d("inputBitmap", bitmap.toString())
        val bitmapScaled = Bitmap.createScaledBitmap(bitmap, 640, 640, true)
        val inputTensor  = TensorImageUtils.bitmapToFloat32Tensor(
            bitmapScaled,
            TensorImageUtils.TORCHVISION_NORM_MEAN_RGB,
            TensorImageUtils.TORCHVISION_NORM_STD_RGB
        )
        Log.d("inputTensor", inputTensor.toString())

        // Ingresar el tensor al modelo y realizar inferencia
        val outputTensor = module?.forward(IValue.from(inputTensor))?.toTuple()?.get(0)?.toTensor()

        // Obtener resultados de la inferencia
        // Obtener resultados de la inferencia
        val scores = outputTensor?.dataAsFloatArray ?: FloatArray(0)
        val bboxes = outputTensor?.dataAsFloatArray ?: FloatArray(0)
        Log.d("ModelPT", ""+ bboxes.toString())

        // Post-procesamiento de los resultados
        val detections = mutableListOf<Detection>()
        val numClasses = 6 // Se asume que solo hay una clase

        for (i in 0 until numClasses) {
            val offset = i * 4 * 100 // Cada clase tiene 100 detecciones
            for (j in 0 until 100) {
                val score = scores[offset + j]
                if (score > 0.98) {
                    val x1 = bboxes[offset + j]
                    val y1 = bboxes[offset + j + 1]
                    val x2 = bboxes[offset + j + 2]
                    val y2 = bboxes[offset + j + 3]

                    val detection = Detection(
                        label = "class",
                        score = score,
                        left = x1,
                        top = y1,
                        right = x2,
                        bottom = y2
                    )

                    detections.add(detection)
                }
            }
            //return outputTensor?.dataAsFloatArray ?: FloatArray(0)
        }

        // Función de Non-Maximum Suppression (NMS)
        fun nms(detections: List<Detection>): List<Detection> {
            // Implementación de NMS
            return detections
        }
        // Aplicar Non-Maximum Suppression (NMS)
        val nmsDetections = nms(detections)
        return nmsDetections



    }
}
// Función de Non-Maximum Suppression (NMS)
fun nms(detections: List<Detection>): List<Detection> {
    // Implementación de NMS
    return detections
}
data class Detection(
    val label: String,
    val score: Float,
    val left: Float,
    val top: Float,
    val right: Float,
    val bottom: Float
)
