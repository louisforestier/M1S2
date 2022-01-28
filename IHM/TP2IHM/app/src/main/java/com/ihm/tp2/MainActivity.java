package com.ihm.tp2;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.os.Bundle;
import android.util.Log;
import android.widget.EditText;

public class MainActivity extends AppCompatActivity {

    private EditText nameEditText;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        Log.i("INFO", "onCreate");
        nameEditText = findViewById(R.id.name_edit_text);
    }

    public void exit(android.view.View v){
        this.finish();
    }

    public void search(android.view.View v){
        Log.i("INFO", "search");
        Intent intent = new Intent(this,InfoPokemon.class);
        intent.putExtra("inputpokemonname",nameEditText.getText().toString());
        startActivity(intent);
    }

    @Override
    protected void onSaveInstanceState(@NonNull Bundle outState) {
        super.onSaveInstanceState(outState);
        Log.i("INFO", "onSaveInstanceState");
        outState.putString("name",nameEditText.getText().toString());
        Log.i("INFO", "name =" + nameEditText.getText().toString());
    }

    @Override
    protected void onRestoreInstanceState(@NonNull Bundle savedInstanceState) {
        super.onRestoreInstanceState(savedInstanceState);
        Log.i("INFO", "onRestoreInstanceState");
        if (savedInstanceState.containsKey("name")){
            String name = savedInstanceState.getString("name");
            nameEditText.setText(name);
        }
    }

    @Override
    protected void onStart() {
        super.onStart();
        Log.i("INFO", "onStart");
    }

    @Override
    protected void onResume() {
        super.onResume();
        Log.i("INFO", "onResume");
    }

    @Override
    protected void onPause() {
        super.onPause();
        Log.i("INFO", "onPause");
    }

    @Override
    protected void onStop() {
        super.onStop();
        Log.i("INFO", "onStop");
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        Log.i("INFO", "onDestroy");
    }

    @Override
    protected void onRestart() {
        super.onRestart();
        Log.i("INFO", "onRestart");
    }

}