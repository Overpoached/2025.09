import { CameraView, useCameraPermissions } from 'expo-camera';
import * as FileSystem from 'expo-file-system';
import { ImageManipulator, SaveFormat } from 'expo-image-manipulator';
import { useRef, useState } from 'react';
import { Alert, Image, Platform, StyleSheet, Text, TouchableOpacity, View } from 'react-native';

const GOOGLE_CLOUD_VISION_API_ENDPOINT = 'https://image-labeler-1074595524961.asia-east1.run.app';

export default function Index() {
  const [permission, requestPermission] = useCameraPermissions();
  const [photo, setPhoto] = useState<string | null>(null);
  const cameraRef = useRef<CameraView>(null);
  
  const labelImage = async (imageUri: string) => {
        try {
            //const resizedPhoto = await ImageManipulator.manipulateAsync(
            //    imageUri,
            //    [{ resize: { width: 500 } }],
            //    { base64: true, compress: 0.8 }
            //);

            let base64code = '';
            if (Platform.OS === 'web') {
              // Web-specific code to read the image as base64
              console.log('Running on web');
              const response = await fetch(imageUri);
              const blob = await response.blob();
              base64code = await new Promise((resolve, reject) => {
                const reader = new FileReader();
                reader.onloadend = () => resolve(reader.result as string);
                reader.onerror = reject;
                reader.readAsDataURL(blob);
              });
              base64code = base64code.replace(/^data:image\/(png|jpeg|jpg);base64,/, ''); // Remove data URL prefix
              //base64code = imageUri.replace(/^data:image\/(png|jpeg|jpg);base64,/, ''); // Remove data URL prefix
            } else {
              base64code = await FileSystem.readAsStringAsync(imageUri, {
                encoding: FileSystem.EncodingType.Base64,
              });
            }

            console.log('Base64 length:', base64code.length, base64code.substring(0, 100) + '...');

            const body = JSON.stringify({
                requests: [
                    {
                        image: {
                            content: base64code
                        }
                    }
                ]
            });
            //console.log('POST Request', body);
            const response = await fetch(
                `${GOOGLE_CLOUD_VISION_API_ENDPOINT}`,
                {
                    method: 'POST',
                    headers: {
                        Accept: 'application/json',
                        'Content-Type': 'application/json'
                    },
                    body: body
                }
            );
            
            console.log('Parsing Response: ', response.status, response.statusText, response.type, 
              response.url, response.headers, response.body);

            if (response.ok) {
                
                const responseJson = await response.json();
                console.log('Google Vision API Response:', responseJson);

                const detectedLabels: string[] = [];
                if (responseJson.result.length > 0) {
                responseJson.result[0].labelAnnotations.forEach((item: any) => {
                  console.log(`Description: ${item.description}, Score: ${item.score}`);
                  detectedLabels.push(item.description + ': ' + item.score.toFixed(5));
                });


                const webLabels: string[] = [];
                //Web detection
                responseJson.result[0].webDetection?.bestGuessLabels.forEach((item: any) => {
                  console.log(`Best Guess: ${item.label}`);
                  webLabels.push(item.label);
                });

                responseJson.result[0].webDetection?.webEntities.forEach((item: any) => {
                  console.log(`Description: ${item.description}, Score: ${item.score}`);
                  webLabels.push(item.description + ': ' + item.score.toFixed(5));
                });

                const alertText: string = detectedLabels.join("\n") + "\nWeb Detection:\n" + webLabels.join("\n");

                alertText.length > 0 ? Alert.alert("Detected Labels", alertText) :
                  Alert.alert("No Labels", "No labels detected.");
              }
              else {
                Alert.alert("No Labels", "No labels detected.");
              }
               
            } else {
                Alert.alert("Error", response.status.toString() || '\n' || response.statusText);
                //setLabels([]);
            }
        } catch (error) {
            console.error(error);
            Alert.alert("Error", "Failed to call Google Vision API.");
            //setLabels([]);
        }
    };


  // Take photo function
  const takePhoto = async () => {
    if (cameraRef.current) {
      try {
        //cameraRef.current._onCameraReady = () => { console.log('Camera is ready'); }; // Ensure camera is ready
        const photocam = await cameraRef.current.takePictureAsync();
        console.log('Photo taken:', photocam.uri);
        console.log(photocam.width, photocam.height, photocam.format);
        const resizedPhotoContext = ImageManipulator.manipulate(photocam.uri);
        if (photocam.height > photocam.width && photocam.height > 1080) {
          resizedPhotoContext.resize({ height: 1080 });
        } else if (photocam.width >= photocam.height && photocam.width > 1080) {
          resizedPhotoContext.resize({ width: 1080 });
        }
        const image = await resizedPhotoContext.renderAsync();
        const result = await image.saveAsync({ format: SaveFormat.JPEG, compress: 0.8 });
        console.log('Resized photo:', result.uri, result.width, result.height, result.base64);
        setPhoto(result.uri);

        await labelImage(result.uri);

        image.release();
        resizedPhotoContext.release();
        
        // Save to local storage
        const filename = `photo-${Date.now()}.jpg`;
        const destination = `${FileSystem.documentDirectory}${filename}`;
        
        //this will invalidate the photo.uri
        // await FileSystem.moveAsync({
        //   from: photo.uri,
        //   to: destination
        // });
        
        // Save to media library
        //await MediaLibrary.saveToLibraryAsync(destination);
      } catch (error) {
        console.error('Error taking photo:', error);
      }
    }
  };

  if (!permission) {
    // Camera permissions are still loading
    return <View />;
  }

  if (!permission.granted) {
    // Camera permissions are not granted yet
    return (
      <View style={styles.container}>
        <Text style={{ textAlign: 'center', paddingBottom: 10 }}>
          We need your permission to show the camera
        </Text>
        <TouchableOpacity style={styles.button} onPress={requestPermission}>
          <Text style={styles.buttonText}>Grant Permission</Text>
        </TouchableOpacity>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      {photo ? (
        // Show captured photo
        <View style={styles.photoContainer}>
          <Image source={{ uri: photo }} style={styles.photo} />
          <TouchableOpacity style={styles.button} onPress={() => setPhoto(null)}>
            <Text style={styles.buttonText}>Take New Photo</Text>
          </TouchableOpacity>
        </View>
      ) : (
        // Show camera
        <CameraView style={styles.camera} facing='back' ref={cameraRef}
          onCameraReady={() => console.log('Camera is ready')}>
          <View style={styles.buttonContainer}>
            <TouchableOpacity style={styles.button} 
                onPress={takePhoto}>
              <Text style={styles.buttonText}>Take Photo</Text>
            </TouchableOpacity>
          </View>
        </CameraView>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
  },
  camera: {
    flex: 1,
  },
  photoContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  photo: {
    width: '90%',
    height: '70%',
    borderRadius: 10,
  },
  buttonContainer: {
    flex: 1,
    flexDirection: 'row',
    backgroundColor: 'transparent',
    margin: 64,
    justifyContent: 'center',
    alignItems: 'flex-end',
  },
  button: {
    backgroundColor: 'white',
    padding: 15,
    borderRadius: 10,
    marginBottom: 20,
  },
  buttonText: {
    fontSize: 16,
    color: 'black',
  },
});