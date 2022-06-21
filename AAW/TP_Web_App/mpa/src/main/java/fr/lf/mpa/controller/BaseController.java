package fr.lf.mpa.controller;

import org.springframework.ui.Model;

public class BaseController {

    private String message = "ALED";
    private String author = "Louis Forestier";
    private String curse =  "AAW";
    private String title = "Multi Page Application";

    public void initModel(Model model) {
        model.addAttribute("message",message);
        model.addAttribute("author",author);
        model.addAttribute("curse",curse);
        model.addAttribute("title",title);
    }
}
