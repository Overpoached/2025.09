
export interface Vertex {
  x: number;
  y: number;
}

export interface BoundingPoly {
  normalizedVertices: Vertex[];
}

export default BoundingPoly;