package com.ihm.tp2;

import androidx.appcompat.app.AppCompatActivity;

import android.os.Bundle;
import android.widget.EditText;
import android.widget.TextView;

public class InfoPokemon extends AppCompatActivity {

    private TextView name;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_info_pokemon);
        name = findViewById(R.id.pokemon_name);
        if (savedInstanceState != null){
            name.setText((String) savedInstanceState.getSerializable("inputpokemonname"));
        }
        else {
            name.setText(getIntent().getStringExtra("inputpokemonname"));
        }
    }
}