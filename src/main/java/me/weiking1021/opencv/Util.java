package me.weiking1021.opencv;

import java.awt.Image;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.File;
import java.io.IOException;

import javax.imageio.ImageIO;
import javax.swing.ImageIcon;
import javax.swing.JFrame;
import javax.swing.JLabel;

import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.highgui.HighGui;

public class Util {

	public static void init() {
		
		nu.pattern.OpenCV.loadShared();
	}
	
	public static Mat image2cvMat(BufferedImage buffered_image) {

		int type;
		
		switch (buffered_image.getType()) {
			// Gray type image
			case BufferedImage.TYPE_BYTE_GRAY:
				type = CvType.CV_8UC1;
			// RGB type image
			case BufferedImage.TYPE_3BYTE_BGR:
				type = CvType.CV_8UC3;				
				break;
			// RGB and alpha type image
			case BufferedImage.TYPE_4BYTE_ABGR:
				type = CvType.CV_8UC4;
				break;
			// Unknown type image
			default: 
				type = -1;
		}
		
		if (type == -1) {
			
			return null; 
		}
		
//		Mat result_mat = new Mat(buffered_image.getWidth(), buffered_image.getHeight(), type);
		Mat result_mat = new Mat(buffered_image.getHeight(), buffered_image.getWidth(), type);
		
		byte[] image_data_array = ((DataBufferByte) buffered_image.getRaster().getDataBuffer()).getData();
		
		result_mat.put(0, 0, image_data_array);
		
		return result_mat;
	}
	
	public static BufferedImage cvMat2image(Mat target_mat) {
		
		return (BufferedImage) HighGui.toBufferedImage(target_mat);
		
		/*int type = BufferedImage.TYPE_BYTE_GRAY;
		
		if (target_mat.channels() > 3) {
			
			type = BufferedImage.TYPE_4BYTE_ABGR;
		}
		else if (target_mat.channels() > 1) {
			
			type = BufferedImage.TYPE_3BYTE_BGR;
		}
		
		int buffer_size = target_mat.channels() * target_mat.cols() * target_mat.rows();
		
		byte[] mat_data_array = new byte[buffer_size];
		
		target_mat.get(0, 0, mat_data_array);
		
		BufferedImage image = new BufferedImage(target_mat.cols(), target_mat.rows(), type);
		
		byte[] image_data_array = ((DataBufferByte) image.getRaster().getDataBuffer()).getData();
		
		System.arraycopy(mat_data_array, 0, image_data_array, 0, mat_data_array.length);
		
		return image;*/
	}
	

	public static void showImage(String title, Mat target_mat) {
		
		showImage(title, cvMat2image(target_mat));
	}
	
	public static void showImage(String title, BufferedImage buffered_image) {
		
		int offset_y = 50;
	    
	    try {
	        
	        JFrame frame = new JFrame(title);
	        
	        /*int w = buffered_image.getWidth();
	        int h = buffered_image.getHeight();*/

	        frame.setSize(512, 512 + offset_y);
	        
	        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE); 
	        
	        ImageIcon image_icon = new ImageIcon(buffered_image.getScaledInstance(512, 512, Image.SCALE_DEFAULT));
	        
	        JLabel label = new JLabel(image_icon);
	        
	        label.setSize(512, 512);
	        
	        label.setLocation(0, offset_y);
	        
	        frame.add(label);
	        
	        frame.setVisible(true);
	    }
	    catch (Exception e) {
	    	
	        e.printStackTrace();
	    }
	}
	
	public static void saveImage(String file_name, BufferedImage buffered_image) throws IOException {
		
		File file = new File(new File("").getAbsoluteFile(), "./opencv/" + file_name + ".png");
		
		ImageIO.write(buffered_image, "png", file);
		
	}
}
