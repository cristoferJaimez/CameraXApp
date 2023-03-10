package com.dev.cris.cameraxapp
import android.Manifest
import android.content.ContentValues
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Color
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.provider.MediaStore
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.ImageCapture
import androidx.camera.video.Recorder
import androidx.camera.video.Recording
import androidx.camera.video.VideoCapture
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.dev.cris.cameraxapp.databinding.ActivityMainBinding
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import android.widget.Toast
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.core.Preview
import androidx.camera.core.CameraSelector
import android.util.Log
import android.view.View
import android.view.WindowInsets
import android.view.WindowManager
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.ImageProxy
import androidx.camera.video.FallbackStrategy
import androidx.camera.video.MediaStoreOutputOptions
import androidx.camera.video.Quality
import androidx.camera.video.QualitySelector
import androidx.camera.video.VideoRecordEvent
import androidx.core.content.FileProvider
import androidx.core.content.PermissionChecker
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.Tensor
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.File
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.text.SimpleDateFormat
import java.util.Locale

typealias LumaListener = (luma: Double) -> Unit


class MainActivity : AppCompatActivity() {
    private lateinit var viewBinding: ActivityMainBinding

    private var imageCapture: ImageCapture? = null
    private lateinit var interpreter: Interpreter
    private lateinit var labels: List<String>

    private lateinit var tflite: Interpreter
    private lateinit var inputImageBuffer: ByteBuffer
    private lateinit var outputBuffer: TensorBuffer

    //private var outputSize: Int = 0
    //private var videoCapture: VideoCapture<Recorder>? = null
    //private var recording: Recording? = null

    private lateinit var cameraExecutor: ExecutorService

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        viewBinding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(viewBinding.root)

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.LOLLIPOP) {
            window.addFlags(WindowManager.LayoutParams.FLAG_DRAWS_SYSTEM_BAR_BACKGROUNDS)
            window.statusBarColor = Color.BLACK
        }



        //cargar modelo
        val model = FileUtil.loadMappedFile(this, "model.tflite")
        val options = Interpreter.Options()
        interpreter = Interpreter(model, options)

        // carga el archivo de etiquetas
        labels = FileUtil.loadLabels(this, "labels.txt")

       
        // Request camera permissions
        if (allPermissionsGranted()) {
            startCamera()
        } else {
            ActivityCompat.requestPermissions(
                this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS)
        }

        // Set up the listeners for take photo and video capture buttons
        viewBinding.imageCaptureButton.setOnClickListener { takePhoto() }
        //viewBinding.videoCaptureButton.setOnClickListener { captureVideo() }

        cameraExecutor = Executors.newSingleThreadExecutor()

        tflite = Interpreter(loadModelFile())


        inputImageBuffer = ByteBuffer.allocateDirect(1 * 256 * 320 * 3 * 4)
        inputImageBuffer.order(ByteOrder.nativeOrder())

        outputBuffer = TensorBuffer.createFixedSize(intArrayOf(1,5040 ,21), DataType.FLOAT32)

    }
    override fun onRequestPermissionsResult(
        requestCode: Int, permissions: Array<String>, grantResults:
        IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQUEST_CODE_PERMISSIONS) {
            if (allPermissionsGranted()) {
                startCamera()
            } else {
                Toast.makeText(this,
                    "Permissions not granted by the user.",
                    Toast.LENGTH_SHORT).show()
                finish()
            }
        }
    }

    // tomar foto
    private fun takePhoto() {
        // Get a stable reference of the modifiable image capture use case
        val imageCapture = imageCapture ?: return


        // Create time stamped name and MediaStore entry.
        val name = SimpleDateFormat(FILENAME_FORMAT, Locale.US)
            .format(System.currentTimeMillis())
        val contentValues = ContentValues().apply {
            put(MediaStore.MediaColumns.DISPLAY_NAME, name)
            put(MediaStore.MediaColumns.MIME_TYPE, "image/jpeg")
            if(Build.VERSION.SDK_INT > Build.VERSION_CODES.P) {
                put(MediaStore.Images.Media.RELATIVE_PATH, "Pictures/Rx-Image")
            }
        }

        // Create output options object which contains file + metadata
        val outputOptions = ImageCapture.OutputFileOptions
            .Builder(contentResolver,
                MediaStore.Images.Media.EXTERNAL_CONTENT_URI,
                contentValues)
            .build()

        // Set up image capture listener, which is triggered after photo has
        // been taken
        imageCapture.takePicture(
            outputOptions,
            ContextCompat.getMainExecutor(this),
            object : ImageCapture.OnImageSavedCallback {
                override fun onError(exc: ImageCaptureException) {
                    Log.e(TAG, "Photo capture failed: ${exc.message}", exc)
                }

                override fun onImageSaved(output: ImageCapture.OutputFileResults){

                    val msg = "Photo capture succeeded: ${output.savedUri!!.path}"
                    Toast.makeText(baseContext, msg, Toast.LENGTH_SHORT).show()
                    Log.d(TAG, msg)


                    val imageUri = output.savedUri
                    val projection = arrayOf(MediaStore.Images.Media.DATA)
                    val cursor = contentResolver.query(imageUri!!, projection, null, null, null)
                    cursor?.moveToFirst()
                    val imagePath = cursor?.getString(cursor.getColumnIndexOrThrow(MediaStore.Images.Media.DATA))
                    val imageFile = File(imagePath)
                    val bitmap = BitmapFactory.decodeFile(imageFile.absolutePath)
                    recognizeImage(bitmap)

                    //val modelByteBuffer = loadModelFile()
                    //val model = Interpreter(modelByteBuffer)
                    //val inputData = FloatArray(1 * 256* 320* 3)
                    //val outputData = runInference(model, inputData)

                }
            }
        )
    }

    //por el momento video no me interesa
    private fun captureVideo() {}


    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener({

            // Used to bind the lifecycle of cameras to the lifecycle owner
            val cameraProvider: ProcessCameraProvider = cameraProviderFuture.get()

            // Preview
            val preview = Preview.Builder()
                .build()
                .also {
                    it.setSurfaceProvider(viewBinding.viewFinder.surfaceProvider)
                }

            imageCapture = ImageCapture.Builder()
                .build()

            val cameraSelector = CameraSelector.Builder()
                .requireLensFacing(CameraSelector.LENS_FACING_BACK)
                .build()

            try {
                cameraProvider.unbindAll()

                cameraProvider.bindToLifecycle(
                    this, cameraSelector, preview, imageCapture
                )

            } catch (exc: Exception) {
                Toast.makeText(this, "Unable to start camera", Toast.LENGTH_SHORT).show()
            }
        }, ContextCompat.getMainExecutor(this))
    }

    //fin de vista previa

    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(
            baseContext, it) == PackageManager.PERMISSION_GRANTED
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
    }

    companion object {
        private const val TAG = "CameraXApp"
        private const val FILENAME_FORMAT = "yyyy-MM-dd-HH-mm-ss-SSS"
        private const val REQUEST_CODE_PERMISSIONS = 10
        private val REQUIRED_PERMISSIONS =
            mutableListOf (
                Manifest.permission.CAMERA,
                Manifest.permission.RECORD_AUDIO
            ).apply {
                if (Build.VERSION.SDK_INT <= Build.VERSION_CODES.P) {
                    add(Manifest.permission.WRITE_EXTERNAL_STORAGE)
                }
            }.toTypedArray()
    }

    private fun getTopLabels(results: FloatArray, labels: List<String>, numLabels: Int = 5): List<String> {
        val sortedResults = results.withIndex().sortedByDescending { it.value }
        return sortedResults.take(numLabels).map { labels[it.index] }
    }

    //imagen a bitmap

    private fun loadModelFile(): MappedByteBuffer {
        val assetFileDescriptor = assets.openFd("model.tflite")
        val inputStream = FileInputStream(assetFileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = assetFileDescriptor.startOffset
        val declaredLength = assetFileDescriptor.declaredLength

        // Create an Interpreter from the model file
        val model = Interpreter(fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength))

        // Print information about the input tensor of the model
        val inputTensorIndex = 0
        val inputTensor = model.getInputTensor(inputTensorIndex)
        val inputTensorName = inputTensor.name()
        val inputTensorShape = inputTensor.shape()
        val inputTensorDataType = inputTensor.dataType()
        println("Model information:")
        println("  Input tensor name: $inputTensorName")
        println("  Input tensor shape: ${inputTensorShape.joinToString()}")
        println("  Input tensor data type: $inputTensorDataType")



        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    private fun runInference(model: Interpreter, inputData: FloatArray): FloatArray {
        // Create an output tensor
        val outputTensorIndex = 0
        val outputTensor = model.getOutputTensor(outputTensorIndex)
        val outputShape = outputTensor.shape()
        val outputDataType = outputTensor.dataType()
        println("  Output tensor shape: ${outputShape.joinToString()}")
        println("  Output tensor data type: $outputDataType")

        // Run inference on the input data
        val outputBuffer = FloatBuffer.allocate(outputTensor.shape().size)
        model.run(inputData, outputBuffer)

        // Convert the output buffer to a float array
        val outputData = FloatArray(outputTensor.shape().size)
        outputBuffer.rewind()
        outputBuffer.get(outputData)

        return outputData
    }

    private fun recognizeImage(bitmap: Bitmap) {
        // Redimensionar el bitmap a 640x640
        val resizedBitmap = Bitmap.createScaledBitmap(bitmap, 256, 320 , true)

        val tensorImage = TensorImage.fromBitmap(resizedBitmap)
        val imageBuffer = tensorImage.buffer


        // Crear un buffer para los valores de píxel
        val pixelBuffer = ByteBuffer.allocateDirect(4 * imageBuffer.capacity())  // 4 bytes por valor float
        pixelBuffer.order(ByteOrder.nativeOrder())  // Usar orden de bytes nativo
        val floatBuffer = pixelBuffer.asFloatBuffer()

        // Copiar los valores de imageBuffer a floatBuffer
        imageBuffer.rewind()
        while (imageBuffer.hasRemaining()) {
            floatBuffer.put(imageBuffer.get().toFloat() / 255.0f)  // Normalizar a valores entre 0 y 1
        }
        //val outputTensorShape = intArrayOf(1, 5040, 21)
        //val outputTensor = Tensor.create(DataType.FLOAT32, outputTensorShape)


        // Ejecutar el modelo
        //val outputBuffers = arrayOf(outputTensor.buffer)



       // interpreter.run(inputBuffer, outputBuffer)
       // var result =  tflite.run(pixelBuffer, null)
        tflite.run(pixelBuffer, outputBuffer.buffer.rewind())
        val result = outputBuffer.floatArray
        Log.d("MODELPROCESS", ""+result.toString())

    }




}