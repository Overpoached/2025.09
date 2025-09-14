import { BoundingPoly } from '../common/Common';
import ImageService from './ImageService';

const GOOGLE_CLOUD_VISION_API_ENDPOINT = 'https://image-labeler-1074595524961.asia-east1.run.app';


export interface LocalizedObjectAnnotation {
  name: string;
  score: number;
  boundingPoly: BoundingPoly;
}

export class ObjectLocalizationResponse {
  private localizedObjects: LocalizedObjectAnnotation[];

  constructor(localizedObjects: LocalizedObjectAnnotation[]) {
    this.localizedObjects = localizedObjects;
  }

  getLocalizedObjects(): LocalizedObjectAnnotation[] {
    return this.localizedObjects;
  }

  getClothingObjects(clothingKeywords: string[]): LocalizedObjectAnnotation[] {
    return this.localizedObjects.filter(object =>
      clothingKeywords.includes(object.name.toLowerCase())
    );
  }

  // You can add more methods here to process or filter the data
  // For example, a method to get objects with a score above a certain threshold
  getObjectsAboveScore(threshold: number): LocalizedObjectAnnotation[] {
    return this.localizedObjects.filter(object => object.score >= threshold);
  }

  print(): void {
    console.log(this.toString());
  }
  toString(): string {
    let result = 'Localized Objects:\n';
    this.localizedObjects.forEach(object => {
      result += `Name: ${object.name}, Score: ${object.score}, BoundingPoly: ${JSON.stringify(object.boundingPoly)}\n`;
    });
    return result;
  }
}

export interface LabelAnnotations {
  description: string;
  score: number;
}
export interface BestGuessLabels {
  label: string;
}
export interface WebEntities {
  description: string;
  score: number;
}
export interface WebDetection {
  bestGuessLabels: BestGuessLabels[];
  webEntities: WebEntities[];
}

export class LabelResponse {
    private labelAnnotations: LabelAnnotations[];
    private webDetection: WebDetection;

    constructor(labelAnnotations: LabelAnnotations[], webDetection: WebDetection) {
        this.labelAnnotations = labelAnnotations;
        this.webDetection = webDetection;
    }

    getLabelAnnotations(): LabelAnnotations[] {
        return this.labelAnnotations;
    }

    getWebDetection(): WebDetection {
        return this.webDetection;
    }

    print(): void {
        console.log(this.toString());
    }
    
    toString(): string {
        let result = 'Label Annotations:\n';
        this.labelAnnotations.forEach(label => {
            result += `Description: ${label.description}, Score: ${label.score}\n`;
        });
        result += 'Web Detection:\n';
        this.webDetection.bestGuessLabels.forEach(label => {
            result += `Best Guess: ${label.label}\n`;
        });
        this.webDetection.webEntities.forEach(entity => {
            result += `Description: ${entity.description}, Score: ${entity.score}\n`;
        });
        return result;
    }
}

class VisionService {

    async createRequest(imageUri: string, features: string[]): Promise<string> {
        const base64code = await ImageService.readImageAsBase64(imageUri);
        return JSON.stringify({
            requests: [
            {
                image: {
                    content: base64code
                },
                features: features,
            }
            ]
        });
    }
    async visionThis(requestJson: string): Promise<Response> {
        const response = await fetch(
            `${GOOGLE_CLOUD_VISION_API_ENDPOINT}`,
            {
                method: 'POST',
                headers: {
                    Accept: 'application/json',
                    'Content-Type': 'application/json'
                },
                body: requestJson
            }
        );

        return response;
    }

    async objectLocalization(imageUri: string): Promise<ObjectLocalizationResponse> {
        
        const body = await this.createRequest(imageUri, ['OBJECT_LOCALIZATION']);
        const response = await this.visionThis(body);

        if (response.ok) {
            const responseJson = await response.json();
            const localizedObjects = responseJson.result[0].localizedObjectAnnotations || [];
            return new ObjectLocalizationResponse(localizedObjects);
        }
        return new ObjectLocalizationResponse([]);
    }

    async labelImage(imageUri: string): Promise<LabelResponse> {
        const base64code = await ImageService.readImageAsBase64(imageUri);

        console.log('Base64 length:', base64code.length, base64code.substring(0, 100) + '...');

        const body = await this.createRequest(imageUri, ['LABEL_DETECTION', 'WEB_DETECTION']);
        const response = await this.visionThis(body);

        //console.log('Parsing Response: ', response.status, response.statusText, response.type,
        //    response.url, response.headers, response.body);

        if (response.ok) {
            const responseJson = await response.json();
            //console.log('Google Vision API Response:', responseJson);
            const labelAnnotations = responseJson.result[0].labelAnnotations || [];
            const webDetection = responseJson.result[0].webDetection || { bestGuessLabels: [], webEntities: [] };

            const labelResponse = new LabelResponse(labelAnnotations, webDetection);
            return labelResponse;
            
        }
        return new LabelResponse([], { bestGuessLabels: [], webEntities: [] });
    }

}

export default new VisionService();