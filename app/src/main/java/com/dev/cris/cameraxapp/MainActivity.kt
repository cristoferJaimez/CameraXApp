package com.dev.cris.cameraxapp
import android.Manifest
import android.content.ContentValues
import android.content.pm.PackageManager
import android.graphics.*
import android.os.Build
import android.os.Bundle
import android.os.Environment
import android.os.Handler
import android.provider.MediaStore
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.ImageCapture
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
import android.view.WindowManager
import android.widget.ImageView
import androidx.camera.core.ImageCaptureException
//import org.tensorflow.lite.DataType
//import org.tensorflow.lite.Interpreter
//import org.tensorflow.lite.support.common.FileUtil
//import org.tensorflow.lite.support.image.TensorImage
//import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.text.SimpleDateFormat
import java.util.Locale


typealias LumaListener = (luma: Double) -> Unit


class MainActivity : AppCompatActivity() {
    private lateinit var viewBinding: ActivityMainBinding

    private var imageCapture: ImageCapture? = null
    //private lateinit var interpreter: Interpreter
    private lateinit var labels: List<String>

    //private lateinit var tflite: Interpreter
    private lateinit var inputImageBuffer: ByteBuffer
    //private lateinit var outputBuffer: TensorBuffer

    private var imageView: ImageView? = null // Declarar una variable para la vista ImageView

    //private var outputSize: Int = 0
    //private var videoCapture: VideoCapture<Recorder>? = null
    //private var recording: Recording? = null

    private lateinit var cameraExecutor: ExecutorService

    private lateinit var pyTorchModel: PyTorchImageSegmentation

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        viewBinding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(viewBinding.root)
        //setContentView(R.layout.activity_main)
        imageView = findViewById(R.id.imageView)


        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.LOLLIPOP) {
            window.addFlags(WindowManager.LayoutParams.FLAG_DRAWS_SYSTEM_BAR_BACKGROUNDS)
            window.statusBarColor = Color.BLACK
        }



        //cargar modelo
        //val model = FileUtil.loadMappedFile(this, "model.tflite")
        //val options = Interpreter.Options()
        //interpreter = Interpreter(model, options)

        // carga el archivo de etiquetas
        //labels = FileUtil.loadLabels(this, "labels.txt")

       //classifier pyTorch
        pyTorchModel = PyTorchImageSegmentation()
        pyTorchModel.assetFilePath(this,"model.torchscript.pt")
        //pyTorchModel.loadModel()

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

        //tflite = Interpreter(loadModelFile())


        inputImageBuffer = ByteBuffer.allocateDirect(1 * 256 * 320 * 3 * 4)
        inputImageBuffer.order(ByteOrder.nativeOrder())

        //outputBuffer = TensorBuffer.createFixedSize(intArrayOf(1,5040 ,21), DataType.FLOAT32)

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
                    // Muestra la imagen resultante en una vista de imagen


                    val imageUri = output.savedUri
                    val projection = arrayOf(MediaStore.Images.Media.DATA)
                    val cursor = contentResolver.query(imageUri!!, projection, null, null, null)
                    cursor?.moveToFirst()
                    val imagePath = cursor?.getString(cursor.getColumnIndexOrThrow(MediaStore.Images.Media.DATA))
                    val imageFile = File(imagePath)
                    val bitmap = BitmapFactory.decodeFile(imageFile.absolutePath)


                    val output = pyTorchModel.inference(bitmap)

                    // Imprimir los resultados de la inferencia
                    Log.d("ModelOutPut", ""+output)
                    // Crear un objeto Paint para dibujar las cajas de detección
                    // Obtener el tamaño de la imagen original
                    val imageWidth = bitmap.width.toFloat()
                    val imageHeight = bitmap.height.toFloat()

                    val paint = Paint()
                    paint.color = Color.BLACK
                    paint.style = Paint.Style.STROKE
                    paint.strokeWidth = 4f

                    val copy = Bitmap.createBitmap(imageWidth.toInt(), imageHeight.toInt(),  bitmap.config)
                    val canvas = Canvas(copy)

                    // Dibujar la imagen original en el canvas
                    canvas.drawBitmap(copy, 0f, 0f, null)


                    for (detection in output) {
                        // Obtener las coordenadas normalizadas de la detección
                        val left = detection.left
                        val top = detection.top
                        val right = detection.right
                        val bottom = detection.bottom

                        // Convertir las coordenadas normalizadas a coordenadas en píxeles
                        val rect = RectF(
                            left * imageWidth,
                            top * imageHeight,
                            right * imageWidth,
                            bottom * imageHeight
                        )

                        // Dibujar el rectángulo en el canvas
                        canvas.drawRect(rect, paint)
                    }

                    // Mostrar el bitmap resultante en un ImageView


                    imageView?.setImageBitmap(copy)
                    Handler().postDelayed({
                        imageView?.setImageBitmap(null)
                    }, 3000)
                    //recognizeImage(bitmap)

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

    /*
    private fun loadModelFile(): MappedByteBuffer {
        val assetFileDescriptor = assets.openFd("model.tflite")
        val inputStream = FileInputStream(assetFileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = assetFileDescriptor.startOffset
        val declaredLength = assetFileDescriptor.declaredLength

        // Create an Interpreter from the model file
        //val model = Interpreter(fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength))

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

        // Carga las etiquetas desde el archivo labels.txt
        val labels = assets.open("labels.txt").bufferedReader().readLines()

        // Ejecuta la inferencia y obtiene los resultados
        tflite.run(pixelBuffer, outputBuffer.buffer.rewind())
        val result = outputBuffer.floatArray


        // Obtiene la forma de la salida del modelo
        val outputShape = tflite.getOutputTensor(0).shape()

        // Obtiene el número de objetos detectados
        val numDetections = outputShape[1].toInt()

        // Obtiene la altura y anchura de la imagen de entrada
        val inputHeight = 256
        val inputWidth = 320

        // Crea un objeto Canvas para dibujar en la imagen original
        // Suponiendo que "bitmap" es el Bitmap que se desea hacer mutable
        // Suponiendo que "bitmap" es el Bitmap que se desea hacer mutable
        val mutableBitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true)


        val canvas = Canvas(mutableBitmap)

        // Crea un objeto Paint para dibujar las cajas delimitadoras y etiquetas
        val paint = Paint()
        paint.color = Color.RED
        paint.style = Paint.Style.STROKE
        paint.strokeWidth = 4f
        paint.textSize = 12f

        // Procesa las coordenadas de las cajas delimitadoras y las etiquetas
        for (i in 0 until numDetections) {
            val offset = i * 4
            val x = result[offset] * inputWidth
            val y = result[offset + 1] * inputHeight
            val width = result[offset + 2] * inputWidth - x
            val height = result[offset + 3] * inputHeight - y

            // Obtiene la etiqueta correspondiente a este objeto
            val labelIndex = result[i * 5 + 4].toInt()
            val label = labels[labelIndex]
            Log.d("MODEL", ""+ label)
            // Dibuja la caja delimitadora en la imagen
            canvas.drawRect(x, y, x + width, y + height, paint)

            // Dibuja la etiqueta junto a la caja delimitadora
            canvas.drawText("$label: ${(result[i * 5] * 100).toInt()}%", x, y - 10, paint)
        }

        // Muestra la imagen resultante en una vista de imagen

        //imageView.setImageBitmap(mutableBitmap)
        imageView?.setImageBitmap(mutableBitmap)
        Handler().postDelayed({
            imageView?.setImageBitmap(null)
        }, 5000)

    }
 */
    private fun saveImageToExternalStorage(bitmap: Bitmap): Boolean {
        val imageFileName = "IMG_${System.currentTimeMillis()}.jpg"
        val storageDir = getExternalFilesDir(Environment.DIRECTORY_PICTURES)

        val imageFile = File(storageDir, imageFileName)
        val outputStream = FileOutputStream(imageFile)
        return try {
            bitmap.compress(Bitmap.CompressFormat.JPEG, 100, outputStream)
            outputStream.flush()
            outputStream.close()
            true
        } catch (e: Exception) {
            false
        }
    }



}