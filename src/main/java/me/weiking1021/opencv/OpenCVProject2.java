package me.weiking1021.opencv;

import java.io.File;

import javax.imageio.ImageIO;

import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

public class OpenCVProject2 {
	
	static {
		
		Util.init();
	}

	public static final File ROOT = new File("").getAbsoluteFile();
	
	private static final int KERNEL_SIZE = 4;
	
	private static final Mat KERNEL = new Mat(new Size(KERNEL_SIZE, KERNEL_SIZE), CvType.CV_8UC1, new Scalar(255));

	public static void main(String args[]) {
		
		for (int i=1; i<=3; i++) {
			
			show("hand_" + i);			
		}
	}

	public static void show(String file_name) {
		
		try {

			Mat bgr_mat = Util.image2cvMat(ImageIO.read(new File(ROOT, "./opencv/" + file_name + ".jpg")));

			Mat hsv_mat = bgr_mat.clone();
			
			Util.bgr2hsv(bgr_mat, hsv_mat);

			Util.showImage(file_name, Util.cvMat2image(bgr_mat));
			
			Mat hsv_skin_mat = findHsvSkin(hsv_mat);
			
//			Util.showImage(file_name + " (Original)", hsv_skin_mat);
			
//			Util.showImage(file_name + " (Close)", close(hsv_skin_mat));
			
			Util.showImage(file_name, writeText(myErosion(myDilation(hsv_skin_mat)), "MyClose"));

//			Util.showImage(file_name + " (Open)", open(hsv_skin_mat));
			
			Util.showImage(file_name, writeText(myDilation(myErosion(hsv_skin_mat)), "MyOpen"));
		}
		catch (Exception e) {

			e.printStackTrace();
		}
	}
	
	private static Mat findHsvSkin(Mat hsv_mat) {

		Mat result = Mat.zeros(hsv_mat.size(), CvType.CV_8UC1);

		for (int i = 0; i < hsv_mat.rows(); i++) {
			for (int j = 0; j < hsv_mat.cols(); j++) {

				byte[] data = new byte[hsv_mat.channels()];

				hsv_mat.get(i, j, data);

				int h = data[0] < 0 ? data[0] + 256 : data[0];
				int s = data[1] < 0 ? data[1] + 256 : data[1];
				int v = data[2] < 0 ? data[2] + 256 : data[2];

				if (0 <= h && h <= 50 && 30 <= s && 90 <= v) {

					result.put(i, j, new byte[] { -1 });
				}
			}
		}	
		
		return result;
	}
	
	@SuppressWarnings("unused")
	private static Mat close(Mat src_mat) {
		
		Mat result = src_mat.clone();
		
		Imgproc.morphologyEx(src_mat, result, Imgproc.MORPH_CLOSE, KERNEL);
		
		return result;
	}
	
	@SuppressWarnings("unused")
	private static Mat open(Mat src_mat) {
		
		Mat result = src_mat.clone();
		
		Imgproc.morphologyEx(src_mat, result, Imgproc.MORPH_OPEN, KERNEL);
		
		return result;		
	}
	
	private static Mat myDilation(Mat src_mat) {
		
		Mat result = Mat.zeros(src_mat.size(), src_mat.type());
		
		int rows = src_mat.rows();
		int cols = src_mat.cols();
		
		int size = (KERNEL_SIZE + 1) / 2;

		for (int i=0; i<rows; i++) {
			for (int j=0; j<cols; j++) {
				
				int row_start = (i < size ? 0 : i - size);
				int row_end = (i + size >= rows ? rows : i + size);
				int col_start = (j < size ? 0 : j - size);
				int col_end = (j + size >= cols ? cols : j + size);
				
				Mat submat = src_mat.submat(row_start, row_end, col_start, col_end);
				
				if (!isZero(submat)) {
					
					result.put(i, j, new byte[] { -1 });
				}
			}
		}
		
		return result;
	}
	
	private static Mat myErosion(Mat src_mat) {

		Mat result = Mat.zeros(src_mat.size(), src_mat.type());
		
		int size = (KERNEL_SIZE + 1) / 2;
		
		int rows = src_mat.rows();
		int cols = src_mat.cols();
		
		for (int i=0; i<rows; i++) {
			for (int j=0; j<cols; j++) {
				
				int row_start = (i < size ? 0 : i - size);
				int row_end = (i + size >= rows ? rows : i + size);
				int col_start = (j < size ? 0 : j - size);
				int col_end = (j + size >= cols ? cols : j + size);
				
				Mat submat = src_mat.submat(row_start, row_end, col_start, col_end);
				
				if (!hasZero(submat)) {
					
					result.put(i, j, new byte[] { -1 });
				}
			}
		}
		
		return result;
	}	
	
	private static boolean isZero(Mat src_mat) {
		
		for (int i=0; i<src_mat.rows(); i++) {
			for (int j=0; j<src_mat.cols(); j++) {
				
				byte[] data = new byte[src_mat.channels()];
				
				src_mat.get(i, j, data);
				
				for (byte b : data) {
					
					if (b != 0) {
							
						return false;
					}
				}
			}
		}
		
		return true;
	}
	
	private static boolean hasZero(Mat src_mat) {
		
		for (int i=0; i<src_mat.rows(); i++) {
			for (int j=0; j<src_mat.cols(); j++) {
				
				byte[] data = new byte[src_mat.channels()];
				
				src_mat.get(i, j, data);
				
				for (byte b : data) {
					
					if (b == 0) {
						
						return true;
					}
				}
			}
		}
		
		return false;
	}
	
	private static Mat writeText(Mat src_mat, String text) {
		
		Imgproc.putText(src_mat, text, new Point(10, 30), 0, 1.0, new Scalar(255));
		
		return src_mat;
	}
}
