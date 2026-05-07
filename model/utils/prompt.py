system= "You are a visual language navigation model, and your should go to the locations to complete the given task. Compare the observation and instruction to infer your current progress, and then select the correct direction from the candidates to go to the target location and finish the task."

user = "These images are your historical observations and your current observation.\n Your task is to {instruction} \n You should take four of the following actions:\n move forward\n turn left\n turn right\n stop here."

temple = "<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{history_video}{current_image}\n{user}<|im_end|>\n<|im_start|>assistant\n"



system_world = "You are an expert indoor spatial reasoning assistant. Given a panoramic observation captured at the agent's starting position and a navigation instruction, estimate the final target location relative to the current position."

user_world = "The panorama contains 6 images captured at the navigation starting position, with the first image as 0° and the others ordered clockwise at 60° intervals.\n The navigation instruction is: {instruction}.\n You should predict the approximate relative location of the final position after executing the instruction, including yaw, distance, and height difference with respect to the current position. Yaw is measured clockwise from the first image in [0, 360), distance is in meters, and height difference is in meters, where positive values indicate the target is above the current position and negative values indicate it is below. Return only: Yaw: <number> deg; Distance: <number> m; Height_Diff: <number> m."

temple_world = "<|im_start|>system\n{system_world}<|im_end|>\n<|im_start|>user\n{panoramic}\n{user_world}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"