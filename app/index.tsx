import { CameraView, useCameraPermissions } from 'expo-camera';
import { useEffect, useRef, useState } from 'react';
import { Alert, Dimensions, FlatList, Image, ScrollView, StyleSheet, Text, TouchableOpacity, View } from 'react-native';
import ImageService from './services/ImageService';
import VisionService, { LocalizedObjectAnnotation } from './services/VisionService';

const screenWidth = Dimensions.get('window').width;
const numColumns = Math.floor(screenWidth / 200); // Calculate number of columns
const itemWidth = (screenWidth - 20) / 3; // Calculate item width with padding

export default function Index() {
  const [permission, requestPermission] = useCameraPermissions();
  const [photo, setPhoto] = useState<string | null>(null);
  const cameraRef = useRef<CameraView>(null);
  const [localizedObjects, setLocalizedObjects] = useState<LocalizedObjectAnnotation[]>([]); // Store localized objects
  const [selectedObjects, setSelectedObjects] = useState<number[]>([]); // Store selected object indices
  const [labelResults, setLabelResults] = useState<{ [key: number]: string }>({}); // Store label results
  const [subImageUris, setSubImageUris] = useState<string[]>([]); // Store sub-image URIs
  const [cameraAspectRatio, setCameraAspectRatio] = useState(1); // Default aspect ratio


  // Take photo function
  const takePhoto = async () => {
    if (cameraRef.current) {
      try {
        const photocam = await cameraRef.current.takePictureAsync();
        console.log('Photo taken:', photocam.uri);

        const resizedUri = await ImageService.resizeImage(photocam.uri);
        console.log('Resized photo:', resizedUri);
        setPhoto(resizedUri);

        //Find relavent objects from photo
        const objects = await VisionService.objectLocalization(resizedUri);
        objects.print();
        setLocalizedObjects(objects.getLocalizedObjects()); // Store localized objects


        // const clothingObjects = objects.getLocalizedObjects();
        // // const clothingObjects = objects.getLocalizedObjects().filter(object =>
        // //   ClothingKeywords.includes(object.name.toLowerCase())
        // // );
        // for (const object of clothingObjects) {
        //   console.log(`Object: ${object.name}, Score: ${object.score}`);

        //   const subImageUri = await ImageService.extractSubImageNative(resizedUri, object.boundingPoly);
        //   console.log(`Sub-image URI: ${subImageUri}`);
        //   // You can further process the sub-image URI as needed
        //   const subImageLabels = await VisionService.labelImage(subImageUri);
        //   subImageLabels.print();
        //   Alert.alert(subImageLabels.toString());
        // }

      } catch (error) {
        console.error('Error taking photo:', error);
      }
    }
  };

  useEffect(() => {
    // Function to get camera aspect ratio (replace with your actual implementation)
    const getCameraAspectRatio = async () => {
        // This is a placeholder - replace with actual code to get camera dimensions
        // For example, you might use cameraRef.current.getSupportedRatiosAsync()
        cameraRef.current?.getAvailablePictureSizesAsync().then(ratios => {
            console.log('Supported camera ratios:', ratios);
        });
        // and then calculate the aspect ratio from the chosen ratio
        // const cameraWidth = 1920; // Example width
        // const cameraHeight = 1080; // Example height
        // const aspectRatio = cameraWidth / cameraHeight;
        // setCameraAspectRatio(aspectRatio);
    };

    getCameraAspectRatio();
  }, []);

  useEffect(() => {
    const generateSubImageUris = async () => {
      if (photo && localizedObjects.length > 0) {
        const uris = await Promise.all(
          localizedObjects.map(async (object) => {
            return await ImageService.extractSubImageNative(photo, object.boundingPoly);
          })
        );
        setSubImageUris(uris);
      }
    };

    generateSubImageUris();
  }, [photo, localizedObjects]);

  const handleSelectObject = (index: number) => {
    if (selectedObjects.includes(index)) {
      setSelectedObjects(selectedObjects.filter(i => i !== index)); // Deselect
    } else {
      setSelectedObjects([...selectedObjects, index]); // Select
    }
  };
  
  const handleLabelSelected = async () => {
    const results: { [key: number]: string } = {};
    if (!photo) {
      console.log('No photo available');
      Alert.alert('No photo available');
      return;
    };
    console.log('Labeling selected objects:', selectedObjects);
    if (selectedObjects.length === 0) {
      Alert.alert('No objects selected', 'Please select at least one object to label.');
      return;
    }
    // For each selected object, extract sub-image and label it
    let logResults = '';
    for (const index of selectedObjects) {
      const object = localizedObjects[index];
      const subImageUri = await ImageService.extractSubImageNative(photo, object.boundingPoly);
      const subImageLabels = await VisionService.labelImage(subImageUri);
      results[index] = subImageLabels.toString();
      console.log(`Labels for object ${index}: ${results[index]}`);
      logResults += results[index] + '\n';
    }
    setLabelResults(results);
    Alert.alert('Labeling Results', logResults);
  };

  const renderItem = ({ item }: { item: { id: string; source: string } }) => (
    <TouchableOpacity
      style={[
        styles.gridItem,
        selectedObjects.includes(Number(item.id)) && styles.selectedGridItem, // Highlight selected items
      ]}
      onPress={() => handleSelectObject(Number(item.id))}
    >
      <Image source={{ uri: item.source }} style={styles.gridImage} resizeMode="cover" />
      
    </TouchableOpacity>

    //{ selectedObjects.includes(index) && <Text style={styles.selectedOverlay}>Selected</Text> }
  );

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
        <ScrollView style={styles.scrollContentContainer}>
          {/* Main Photo Section */}
          {/* <View style={styles.photoContainer}>
            <Image source={{ uri: photo }} style={styles.photo} resizeMode="cover"/>
          </View> */}

          {/* Separator Line */}
          {/* <View style={styles.separator} /> */}

          <FlatList
            data={subImageUris.map((uri, index) => ({
              id: index.toString(),
              source: uri, // Generate sub-image URIs
            }))}
            renderItem={renderItem}
            keyExtractor={(item) => item.id}
            numColumns={3} // Dynamic number of columns
            scrollEnabled={false} // Disable FlatList scrolling so it works with the parent ScrollView
            ListHeaderComponent={() => (
              <>
                <View style={styles.photoContainer}>
                  <Image source={{ uri: photo }} style={styles.photo} resizeMode="cover"/>
                </View>
                <View style={styles.separator} />
              </>
            )}
            style={styles.gridContainer}
          />
          {/* <View style={styles.gridContainer}>
            {subImageUris.map((uri, index) => (
              renderItem(uri, index)
            ))}
          </View> */}

          <TouchableOpacity style={styles.button} onPress={handleLabelSelected}>
            <Text style={styles.buttonText}>Label Selected</Text>
          </TouchableOpacity>

          <TouchableOpacity style={styles.button} onPress={() => setPhoto(null)}>
            <Text style={styles.buttonText}>Take New Photo</Text>
          </TouchableOpacity>
        </ScrollView>
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
    //justifyContent: 'center',
  },
  scrollContentContainer: {
    flexGrow: 1,
  },
  camera: {
    flex: 1,
  },
  photoContainer: {
    width: '100%',
    height: 300,
    padding: 10,
  },
  photo: {
    width: '100%',
    height: '100%',
    borderRadius: 10,
  },
  separator: {
    width: '100%',
    height: 1,
    backgroundColor: '#e0e0e0',
    //marginVertical: 10,
  },
  buttonContainer: {
    position: 'absolute',
    left: 0,
    right: 0,
    bottom: 20,
    //flex: 1,
    flexDirection: 'row',
    backgroundColor: 'transparent',
    //margin: 64,
    justifyContent: 'center',
    //alignItems: 'flex-end',
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
  gridContainer: {
    //marginTop: 20,
    //paddingHorizontal: 10, // Add horizontal padding
    flexGrow: 1,
    //flexDirection: 'row', // Enable horizontal layout
    //flexWrap: 'wrap', // Allow items to wrap to the next line
    //justifyContent: 'flex-start', // Align items to the start of the container
  },
  gridItem: {
    margin: 5,
    aspectRatio: 1, // Ensure square aspect ratio
    borderWidth: 2,
    borderColor: 'transparent',
    //minWidth: 200, // Minimum width for each item
    width: itemWidth, // Set a fixed width for each item
    height: itemWidth,
    //maxWidth: itemWidth,
    //minWidth: itemWidth,
  },
  selectedGridItem: {
    borderColor: 'blue', // Highlight selected items
  },
  gridImage: {
    //flex: 1,
    //resizeMode: 'cover',
    width: '100%',
    height: '100%',
    aspectRatio: 1,
  },
  selectedOverlay: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    backgroundColor: 'rgba(0, 0, 255, 0.3)', // Semi-transparent blue overlay
    justifyContent: 'center',
    alignItems: 'center',
    color: 'white',
    fontSize: 16,
  },
});