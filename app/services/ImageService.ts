import * as FileSystem from 'expo-file-system';
import { Image } from 'expo-image';
import { ImageManipulator, SaveFormat } from 'expo-image-manipulator';
import { Platform } from 'react-native';
import { BoundingPoly } from '../common/Common';

class ImageService {

    // Get a copy of resized image
  async resizeImage(uri: string): Promise<string> {
    const imageRef = await Image.loadAsync(uri);
    if (!imageRef) {
      throw new Error('Failed to load image');
    }
    const resizedPhotoContext = ImageManipulator.manipulate(imageRef);
    const { width, height } = imageRef

    if (height > width && height > 1080) {
      resizedPhotoContext.resize({ height: 1080 });
    } else if (width >= height && width > 1080) {
      resizedPhotoContext.resize({ width: 1080 });
    }

    const image = await resizedPhotoContext.renderAsync();
    const result = await image.saveAsync({ format: SaveFormat.JPEG, compress: 0.8 });

    image.release();
    resizedPhotoContext.release();
    imageRef.release();

    return result.uri;
  }

  async extractSubImageNative(imageUri: string, boundingPoly: BoundingPoly): Promise<string> {
    const x = boundingPoly.normalizedVertices[0].x;
    const y = boundingPoly.normalizedVertices[0].y;
    const width = boundingPoly.normalizedVertices[2].x - boundingPoly.normalizedVertices[0].x;
    const height = boundingPoly.normalizedVertices[2].y - boundingPoly.normalizedVertices[0].y;

    const imageRef = await Image.loadAsync(imageUri);
    if (!imageRef) {
      throw new Error('Failed to load image');
    }
    const { width: originalWidth, height: originalHeight } = imageRef;

    const cropData = {
      originX: x * originalWidth,
      originY: y * originalHeight,
      width: width * originalWidth,
      height: height * originalHeight,
    };

    const context = await ImageManipulator.manipulate(imageRef);
    context.crop(cropData);
    const image = await context.renderAsync();
    const result = await image.saveAsync({ format: SaveFormat.JPEG, compress: 0.8 });

    image.release();
    context.release();
    imageRef.release();

    return result.uri;
  }

  async readImageAsBase64(imageUri: string): Promise<string> {
    if (Platform.OS === 'web') {
      console.log('Running on web');
      const response = await fetch(imageUri);
      const blob = await response.blob();
      return new Promise<string>((resolve, reject) => {
        const reader = new FileReader();
        reader.onloadend = () => resolve(reader.result as string);
        reader.onerror = reject;
        reader.readAsDataURL(blob);
      }).then(base64code => base64code.replace(/^data:image\/(png|jpeg|jpg);base64,/, ''));
    } else {
      console.log('Running on native');
      return FileSystem.readAsStringAsync(imageUri, {
        encoding: FileSystem.EncodingType.Base64,
      });
    }
  }
}

export default new ImageService();