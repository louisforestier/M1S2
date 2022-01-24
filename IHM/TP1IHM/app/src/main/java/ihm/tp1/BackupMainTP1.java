package ihm.tp1;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Context;
import android.os.Build;
import android.os.Bundle;
import android.os.VibrationEffect;
import android.os.Vibrator;
import android.widget.EditText;
import android.widget.RadioButton;
import android.widget.TextView;
import android.widget.Toast;

import ihm.tp2.R;

public class BackupMainTP1 extends AppCompatActivity {

    public static String TAG = "UPConvTemp"; // Identifiant pour les messages de log
    private EditText editInputTemp; // Boite de saisie de la température
    private RadioButton rbCelsius; // Bouton radio indiquant si la saisie est en Celsius
    private RadioButton rbFahrenheit; // Bouton radio indiquant si la saisie est en Fahrenheit
    private TextView dispResult; // Le TextView qui affichera le résultat

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        editInputTemp = findViewById(R.id.inputtext);
        dispResult = findViewById(R.id.resultText);
        rbCelsius = findViewById(R.id.rbCelsius);
        rbFahrenheit = findViewById(R.id.rbFahrenheit);
    }

    public void vibrate(long duration_ms){
        Vibrator v = (Vibrator) getSystemService(Context.VIBRATOR_SERVICE);
        if (duration_ms < 1)
            duration_ms = 1;
        if (v != null && v.hasVibrator()){
            if (Build.VERSION.SDK_INT >= 26){
                v.vibrate(VibrationEffect.createOneShot(duration_ms, VibrationEffect.DEFAULT_AMPLITUDE));
            }
            else {
                v.vibrate(duration_ms);
            }
        }
    }

    public void toast(String msg){
        Toast.makeText(this,msg,Toast.LENGTH_SHORT).show();
    }

    public void action_convert(android.view.View v){
        String text = editInputTemp.getText().toString();
        Double temp = Double.parseDouble(text);
        String unit;
        if (rbCelsius.isChecked()){
            temp = (temp*9/5) + 32;
            unit = "°F";
        } else {
            temp = (temp-32)*5/9;
            unit = "°C";
        }
        temp = Math.round(temp * 10) / 10.0;
        dispResult.setText(temp + unit);

        toast(dispResult.getText().toString()
        );
    }
}