package fr.lf.mpa.controller;

import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;

@Controller
public class IndexController extends BaseController{

    @GetMapping(value = {"/","/index"})
    public String showIndexPage(Model model) {
        initModel(model);
        return "index";
    }
}
