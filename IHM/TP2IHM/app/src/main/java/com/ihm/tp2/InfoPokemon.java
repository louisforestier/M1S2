package com.ihm.tp2;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.AsyncTask;
import android.os.Bundle;
import android.util.Log;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import org.jsoup.Jsoup;
import org.jsoup.nodes.Document;
import org.jsoup.nodes.Element;
import org.jsoup.select.Elements;

import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.stream.Collectors;


public class InfoPokemon extends AppCompatActivity {

    class PokeRequest extends AsyncTask<Void,Integer,Void> {
        private String name;
        private String restype;
        private String resname;
        private String resweight;
        private String ressize;
        private String resimg;

        public PokeRequest(String name) {
            this.name = name;
            restype = "<none>";
            resname = "<none>";
            resweight = "<none>";
            ressize = "<none>";
        }

        @Override
        protected Void doInBackground(Void... voids) { // Se fait en background du thread UI
            try {
                Document doc = Jsoup.connect("https://www.pokepedia.fr/" + name).get();
                Element tableinfo = doc.selectFirst("table.tableaustandard");

                Element img = tableinfo.select("img").first();
                resimg = "https://www.pokepedia.fr" + img.attr("src");
                Elements names = tableinfo.select("th.entêtesection");
                for (Element e : names) {
                    resname = e.ownText();
                    Log.v(MainActivity.APP_TAG,"Entete section: " + resname);
                }

                Log.v(MainActivity.APP_TAG,"=====>>>>>  FINAL Entete section: " + resname);

                Elements rows = tableinfo.select("tr");
                for (Element row : rows) {
                    Log.v(MainActivity.APP_TAG,"=====>>>>>  new line. ");
                    if(row.select("a[title*=taille]").size() > 0) {
                        Element target = row.selectFirst("td");
                        if(target != null) {
                            ressize = target.ownText();
                            Log.v(MainActivity.APP_TAG,"=====>>>>>  Find a size: " + ressize);
                        }
                        else
                            Toast.makeText(InfoPokemon.this,R.string.error_no_dom_entity, Toast.LENGTH_LONG).show();
                    }

                    if(row.select("a[title*=poids]").size() > 0) {
                        Element target = row.selectFirst("td");
                        if(target != null) {
                            resweight = target.ownText();
                            Log.v(MainActivity.APP_TAG,"=====>>>>>  Find a weight: " + resweight);
                        }
                        else
                            Toast.makeText(InfoPokemon.this,R.string.error_no_dom_entity, Toast.LENGTH_LONG).show();
                    }

                }


                Elements elems = tableinfo.select("a[title*=type]");
                ArrayList<String> types = new ArrayList<>();
                for (Element e: elems) {
                    if(!e.attr("title").equalsIgnoreCase("Type")) {
                        String rawtype = e.attr("title");
                        String type = rawtype.replace(" (type)","");
                        types.add(type);
                        Log.v(MainActivity.APP_TAG,"\tFind type: " +type);
                    }
                }
                restype = types.stream().collect(Collectors.joining(" - "));
            } catch (IOException e) {
                Log.e(MainActivity.APP_TAG,"Error during connection...",e);
                // e.printStackTrace();
            }

            return null;
        }

        @Override
        protected void onProgressUpdate(Integer... values) {
            // Inutile ici, cf doc
            super.onProgressUpdate(values);
        }

        @Override
        protected void onPostExecute(Void aVoid) { // S'exécute sur le ThreadUI après doInBackground
            super.onPostExecute(aVoid);
            // ATTENTION, il faut adapter le code ci-dessous avec vos controles graphiques.
            InfoPokemon.this.pokemon_name.setText(resname);
            InfoPokemon.this.pokemon_type.setText(restype);
            InfoPokemon.this.pokemon_size.setText(ressize);
            InfoPokemon.this.pokemon_weight.setText(resweight);
            new DownloadImageTask(InfoPokemon.this.pokemon_img).execute(resimg);
            Toast.makeText(InfoPokemon.this, R.string.end_request, Toast.LENGTH_SHORT).show();

            // c'est ici que vous devrez ajouter l'écriture de votre fichier en FIN de sujet!!!
        }
    }

    private class DownloadImageTask extends AsyncTask<String, Void, Bitmap> {
        ImageView bmImage;

        public DownloadImageTask(ImageView bmImage) {
            this.bmImage = bmImage;
        }

        protected Bitmap doInBackground(String... urls) {
            String urldisplay = urls[0];
            Log.i(MainActivity.APP_TAG,urldisplay);
            Bitmap mIcon11 = null;
            try {
                InputStream in = new java.net.URL(urldisplay).openStream();
                mIcon11 = BitmapFactory.decodeStream(in);
            } catch (Exception e) {
                Log.e("Error", e.getMessage());
                e.printStackTrace();
            }
            return mIcon11;
        }

        protected void onPostExecute(Bitmap result) {
            bmImage.setImageBitmap(result);
        }
    }

    private TextView pokemon_name;
    private TextView pokemon_type;
    private TextView pokemon_size;
    private TextView pokemon_weight;
    private ImageView pokemon_img;
    private Button browserButton;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        setContentView(R.layout.activity_info_pokemon);
        browserButton = findViewById(R.id.browser_button);
        pokemon_name = findViewById(R.id.pokemon_name);
        pokemon_type = findViewById(R.id.pokemon_type);
        pokemon_weight = findViewById(R.id.pokemon_weight);
        pokemon_size = findViewById(R.id.pokemon_height);
        pokemon_img = findViewById(R.id.imageView);
        if (savedInstanceState != null){
            pokemon_name.setText((String) savedInstanceState.getSerializable("inputpokemonname"));
        }
        else {
            pokemon_name.setText(getIntent().getStringExtra("inputpokemonname"));
        }

        new PokeRequest(pokemon_name.getText().toString()).execute();

    }

    public void openBrowser(android.view.View v){
        Log.i("INFO", "openBrowser");
        String url = "https://www.pokepedia.fr/"+ pokemon_name.getText().toString();
        Intent intent = new Intent(Intent.ACTION_VIEW);
        intent.setData(Uri.parse(url));
        startActivity(intent);
    }
}