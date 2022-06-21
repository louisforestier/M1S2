package fr.algo3d;

import fr.algo3d.controller.MainPaneController;
import fr.algo3d.model.Scene;
import javafx.application.Application;
import javafx.fxml.FXMLLoader;
import javafx.scene.Parent;
import javafx.scene.image.Image;
import javafx.scene.image.PixelWriter;
import javafx.scene.image.WritableImage;
import javafx.scene.image.WritablePixelFormat;
import javafx.stage.Stage;

import java.io.*;

/**
 *
 * @author P. Meseure based on a Java Adaptation of a C code by B. Debouchages (M1, 2018-2019)
 */
public class Main extends Application
{
    /**
     * 
     * @param fout : output file stream
     * @param n : short to write to disc in little endian
     */
    private static void writeShort(FileOutputStream fout,int n) throws IOException
    {
        fout.write(n&255);
        fout.write((n>>8)&255);
    }

    /**
     * 
     * @param filename name of final TGA file
     * @param buffer buffer that contains the image. 3 bytes per pixel ordered this way : Blue, Green, Red
     * @param width Width of the image
     * @param height Height of the image
     * @throws FileNotFoundException
     * @throws UnsupportedEncodingException
     * @throws IOException 
     */
    private static void saveTGA(String filename, byte buffer[], int width, int height) throws IOException, UnsupportedEncodingException {

        FileOutputStream fout = new FileOutputStream(new File(filename));

        fout.write(0); // Comment size, no comment
        fout.write(0); // Colormap type: No colormap
        fout.write(2); // Image type
        writeShort(fout,0); // Origin
        writeShort(fout,0); // Length
        fout.write(0); // Depth
        writeShort(fout,0); // X origin
        writeShort(fout,0); // Y origin
        writeShort(fout,width); // Width of the image
        writeShort(fout,height); // Height of the image
        fout.write(24); // Pixel size in bits (24bpp)
        fout.write(0); // Descriptor

        /* Write the buffer */
        fout.write(buffer);

        fout.close();
    }


    static long chrono = 0 ;

    static void startChrono() {
        chrono = java.lang.System.currentTimeMillis() ;
    }

    static void stopChrono() {
        long chrono2 = java.lang.System.currentTimeMillis() ;
        long temps = chrono2 - chrono ;
        System.out.println("Temps ecoule = " + temps + " ms") ;
    }

    /**
     * @param args no command line arguments
     */
    public static void main(String[] args) {
        launch(args);
    }

    @Override
    public void start(Stage primaryStage) throws Exception {
        int w=1920;
        int h=1080;
        byte buffer[]=new byte[3*w*h];
        byte image[]=new byte[3*w*h];
        Scene scene = new Scene();
        startChrono();
        scene.renderParallelNestedLoops(w, h, buffer, image);
        stopChrono();
        try {
            saveTGA("imagetest.tga",buffer,w,h);
        }
        catch(Exception e)
        {
            System.err.println("TGA file not created :"+e);
        }

        FXMLLoader loader = new FXMLLoader(getClass().getResource("view/MainPane.fxml"));
        Parent root = loader.load();
        MainPaneController controller = loader.getController();
        controller.setBackground(getFXImage(image,w,h));
        primaryStage.setScene(new javafx.scene.Scene(root,1280,720));
        primaryStage.setTitle("Ray Tracing Project");
        primaryStage.show();
    }


    public static Image getFXImage(byte[] buffer, int width, int height) {
        WritableImage image = new WritableImage(width,height);
        PixelWriter writer = image.getPixelWriter();
        writer.setPixels(0,0,width,height, WritablePixelFormat.getByteRgbInstance(),buffer,0,width*3);
        return image;
    }
}

