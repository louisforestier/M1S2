package fr.algo3d.controller;

import javafx.fxml.FXML;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import javafx.scene.layout.*;

public class MainPaneController {
    @FXML
    private AnchorPane mainPane;

    @FXML
    private ImageView imageView;

    @FXML
    public void initialize(){
        imageView.fitHeightProperty().bind(mainPane.heightProperty());
        imageView.fitWidthProperty().bind(mainPane.widthProperty());
    }

    public void setBackground(Image image) {
        imageView.setImage(image);
    }

    public void setMainPaneBackground(Image image){
        BackgroundImage backgroundImage = new BackgroundImage(image, BackgroundRepeat.NO_REPEAT,BackgroundRepeat.NO_REPEAT, BackgroundPosition.CENTER,BackgroundSize.DEFAULT);
        Background bg = new Background(backgroundImage);
        mainPane.setBackground(bg);
    }

}
