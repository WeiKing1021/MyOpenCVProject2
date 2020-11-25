package me.weiking1021.opencv;

import java.io.File;

import javax.imageio.ImageIO;

import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

public class OpenCVProject2 {
	
	public static final File ROOT = new File("").getAbsoluteFile();

	public static Mat m;
	
	public static void main(String args[]) {
		
		Util.init();
		
		try {
			
			Mat hand_mat_1 = Util.image2cvMat(ImageIO.read(new File(ROOT, "./opencv/hand_7.jpg")));
			
			Mat hand_hsvmat_1 = hand_mat_1.clone();
			
			m = hand_hsvmat_1;
			
			Imgproc.cvtColor(hand_mat_1, hand_hsvmat_1, Imgproc.COLOR_BGR2HSV);
			
			Util.showImage("Hand 1", Util.cvMat2image(hand_mat_1));
			
//			Util.showImage("Hand hsv 1", Util.cvMat2image(hand_hsvmat_1));
			
			Mat result = Mat.zeros(hand_hsvmat_1.size(), CvType.CV_8UC1);
			
			for (int i=0; i<hand_hsvmat_1.rows(); i++) {
				for (int j=0; j<hand_hsvmat_1.cols(); j++) {
				
					byte[] data = new byte[hand_hsvmat_1.channels()];
					
					hand_hsvmat_1.get(i, j, data);
					
					int h = data[0] < 0 ? data[0] + 256 : data[0];
					int s = data[1] < 0 ? data[1] + 256 : data[1];
					int v = data[2] < 0 ? data[2] + 256 : data[2];

					if (0 <= h && h <= 30 && 20 <= s && 120 <= v) {
						
						result.put(i, j, new byte[] {-1});
					}					
					
					if (0 <= h && h <= 30 && 100 <= s && 30 <= v) {
					
						result.put(i, j, new byte[] {-1});
					}
				}				
			}
			
//			Util.showImage("Result", Util.cvMat2image(result));
			
			Mat kernel = new Mat(new Size(5, 5), CvType.CV_8UC1, new Scalar(255));

			Mat close_after  = result.clone();
			
			Imgproc.morphologyEx(result, close_after, Imgproc.MORPH_CLOSE, kernel);
			
			Util.showImage("Close", close_after);
		}	
		catch (Exception e) {
			
			e.printStackTrace();
		}
	}
}
