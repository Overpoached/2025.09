#pragma once

#include <string>

namespace ohy {

	/// <summary>
	/// Represents the main engine for handling image processing and car model identification.
	/// </summary>
	class Engine {
	public:
		~Engine();
		/// <summary>
		/// Initializes the engine with the specified model path.
		/// </summary>
		/// <param name="modelPath">The path to the model file.</param>
		void Initialize(const std::string& modelPath);
		void Shutdown();
		/// <summary>
		/// Processes the input image and returns the identified car model.
		/// </summary>
		/// <param name="imagePath">The path to the input image.</param>
		/// <returns>The identified car model as a string.</returns>
		std::string ProcessImage(const std::string& imagePath);

	private:
		bool isInitialized = false;  // Flag to check if the engine is initialized
	};
}