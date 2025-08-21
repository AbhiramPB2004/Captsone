from moviepy.editor import VideoFileClip

# Load a short video and extract the first 5 seconds to a new file
input_path = "input_video.mp4"    # Replace with the path to a small video you have
output_path = "output_clip.mp4"

# Create a 5-second subclip and save it
clip = VideoFileClip(input_path).subclip(0, 5)
clip.write_videofile(output_path)

print("MoviePy test complete! Saved output_clip.mp4.")
