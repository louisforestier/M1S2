package com.ihm.tp2;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;

import android.Manifest;
import android.app.Activity;
import android.content.Intent;
import android.content.SharedPreferences;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.os.Environment;
import android.preference.PreferenceManager;
import android.util.Log;
import android.widget.EditText;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.util.Date;
import java.util.Set;
import java.util.TreeSet;

public class MainActivity extends AppCompatActivity {

    public static final String APP_TAG = "POKESTAT";
    private EditText nameEditText;
    private Set<String> searchedPokemonName;
    // Listes des permissions
    private static final int REQUEST_EXTERNAL_STORAGE = 1;
    private static String[] PERMISSIONS_STORAGE = {
            Manifest.permission.READ_EXTERNAL_STORAGE,
            Manifest.permission.WRITE_EXTERNAL_STORAGE
    };
    public static void verifyStoragePermissions(Activity activity) {
// Vérifie si nous avons les droits d'écriture
        int permission = ActivityCompat.checkSelfPermission(activity,
                Manifest.permission.WRITE_EXTERNAL_STORAGE);
        if (permission != PackageManager.PERMISSION_GRANTED) {
// Aie, il faut les demander àl'utilisateur
            ActivityCompat.requestPermissions(
                    activity,
                    PERMISSIONS_STORAGE,
                    REQUEST_EXTERNAL_STORAGE
            );
        }
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        Log.i("INFO", "onCreate");
        nameEditText = findViewById(R.id.name_edit_text);
        this.reload_historic();
        this.display_historic();
        verifyStoragePermissions(this);
    }

    // Fonction qui recharge un historique
    public void reload_historic() {
        // Récuperation de l'objet unique qui s'occupe de la sauvegarde
        SharedPreferences sharedPref = PreferenceManager.getDefaultSharedPreferences(this);
        // Récuperation de l'ancienne valeur ou d'une valeur par défaut
        searchedPokemonName = sharedPref.getStringSet("historyPokemonName", new TreeSet<String>());
    }

    // Fonction qui affiche l'historique àpartir de l'attribut searchedPokemonName
    // Il faut donc avoir chargé l'historique avant!
    public void display_historic() {
        Log.d(MainActivity.APP_TAG, "Historique (" + (new Date()) + ") size= " + searchedPokemonName.size() + ": ");
        for (String item : searchedPokemonName) {
            Log.d(MainActivity.APP_TAG, "\t- " + item);
        }
    }

    public void write_historic_in_file(android.view.View v) {
        File folder = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS);
        File fileout = new File(folder, "pokestat_historic.txt");
        try (FileOutputStream fos = new FileOutputStream(fileout)) {
            PrintStream ps = new PrintStream(fos);
            ps.println("Start of my historic");
            for (String item : searchedPokemonName) {
                ps.println("\t- " + item);
            }
            ps.close();
        } catch (FileNotFoundException e) {
            Log.e(APP_TAG,"File not found",e);
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
            Log.e(APP_TAG,"Error I/O",e);
        }
    }

    public void exit(android.view.View v){
        this.finish();
    }

    public void search(android.view.View v){
        Log.i("INFO", "search");
        this.searchedPokemonName.add(nameEditText.getText().toString());
        SharedPreferences sharedPref = PreferenceManager.getDefaultSharedPreferences(this);
        sharedPref.edit().putStringSet("historyPokemonName", searchedPokemonName);
        sharedPref.edit().commit();
        this.display_historic();
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