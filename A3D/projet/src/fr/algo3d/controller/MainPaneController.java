package fr.algo3d.controller;

import javafx.fxml.FXML;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import javafx.scene.layout.*;

/**
 * Controller for the main pane.
 */
public class MainPaneController {
    /**
     * Main pane.
     */
    @FXML
    private AnchorPane mainPane;

    /**
     * ImageView to contain the rendered image.
     */
    @FXML
    private ImageView imageView;

    /**
     * Binds the ImageView and the main pane heights and widths.
     */
    @FXML
    public void initialize(){
        imageView.fitHeightProperty().bind(mainPane.heightProperty());
        imageView.fitWidthProperty().bind(mainPane.widthProperty());
    }

    /**
     * Set the image of the ImageView to the parameter.
     * @param image
     */
    public void setBackground(Image image) {
        imageView.setImage(image);
    }

}
