package ihm.tp2;

import android.content.Context;
import android.os.Build;
import android.os.Bundle;
import android.os.VibrationEffect;
import android.os.Vibrator;
import android.util.Log;
import android.view.View;
import android.widget.AdapterView;
import android.widget.EditText;
import android.widget.Spinner;
import android.widget.Switch;
import android.widget.TextView;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;

import ihm.tp2.R;

public class MainActivity extends AppCompatActivity {

    public static String TAG = "UPConvTemp"; // Identifiant pour les messages de log
    private EditText editInputTemp; // Boite de saisie de la température
    private TextView dispResult; // Le TextView qui affichera le résultat
    private Spinner inputUnit;
    private Spinner outputUnit;
    private Switch autoConvertSwitch;
    private boolean autoConvert = false;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        setContentView(R.layout.activity_main);
        editInputTemp = findViewById(R.id.inputtext);
        dispResult = findViewById(R.id.resultText);
        inputUnit = findViewById(R.id.input_unit);
        outputUnit = findViewById(R.id.output_unit);
        autoConvertSwitch = findViewById(R.id.auto_switch);
        editInputTemp.setOnKeyListener((v, keyCode, event) -> {
            Log.d("INFO","OnKeyListener");
            if (autoConvert) {
                String text = editInputTemp.getText().toString();
                Double temp = Double.parseDouble(text);
                convertTemp(temp);
            }
            return false;
        });
        inputUnit.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> parent, View view, int position, long id) {
                Log.d("INFO","Input: OnItemSelected");
                if (autoConvert) {
                    String text = editInputTemp.getText().toString();
                    Double temp = Double.parseDouble(text);
                    convertTemp(temp);
                }
            }

            @Override
            public void onNothingSelected(AdapterView<?> parent) {
                Log.d("INFO","Input OnNothingSelected");
                if (autoConvert)
                    toast(getString(R.string.no_unit));
            }
        });
        outputUnit.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> parent, View view, int position, long id) {
                Log.d("INFO","Output: OnItemSelected");
                if (autoConvert) {
                    String text = editInputTemp.getText().toString();
                    Double temp = Double.parseDouble(text);
                    convertTemp(temp);
                }
            }

            @Override
            public void onNothingSelected(AdapterView<?> parent) {
                Log.d("INFO", "Output: OnNothingSelected");
                if (autoConvert) {
                    toast(getString(R.string.no_unit));
                }
            }
        });
        autoConvertSwitch.setOnCheckedChangeListener((buttonView, isChecked) -> {
            autoConvert = !autoConvert;
            Log.d("INFO", "Switch: OnCheckedChangeListener");
            if (autoConvert) {
                String text = editInputTemp.getText().toString();
                Double temp = Double.parseDouble(text);
                convertTemp(temp);
            }
        });
    }

    public void vibrate(long duration_ms) {
        Vibrator v = (Vibrator) getSystemService(Context.VIBRATOR_SERVICE);
        if (duration_ms < 1)
            duration_ms = 1;
        if (v != null && v.hasVibrator()) {
            if (Build.VERSION.SDK_INT >= 26) {
                v.vibrate(VibrationEffect.createOneShot(duration_ms, VibrationEffect.DEFAULT_AMPLITUDE));
            } else {
                v.vibrate(duration_ms);
            }
        }
    }

    public void toast(String msg) {
        Toast.makeText(this, msg, Toast.LENGTH_SHORT).show();
    }

    public void convertTemp(Double temp) {
        Double tmp = temp;
        switch (inputUnit.getSelectedItemPosition()) {
            case 0:
                switch (outputUnit.getSelectedItemPosition()) {
                    case 0:
                        break;
                    case 1:
                        tmp = (tmp * 9 / 5) + 32;
                        break;
                    case 2:
                        tmp = tmp + 273.15;
                        break;
                }
                break;
            case 1:
                switch (outputUnit.getSelectedItemPosition()) {
                    case 0:
                        tmp = (tmp - 32) * 5 / 9;
                        break;
                    case 1:
                        break;
                    case 2:
                        tmp = (tmp + 459.67) / 1.8;
                        break;
                }
                break;
            case 2:
                switch (outputUnit.getSelectedItemPosition()) {
                    case 0:
                        tmp = tmp - 273.15;
                        break;
                    case 1:
                        tmp = tmp * 1.8 - 459.67;
                        break;
                    case 2:
                        break;
                }
                break;
        }
        tmp = Math.round(tmp * 10) / 10.0;
        dispResult.setText(tmp.toString());

        toast(dispResult.getText().toString()
        );
    }
    public void action_convert(android.view.View v){
        String text = editInputTemp.getText().toString();
        Double temp = Double.parseDouble(text);
        convertTemp(temp);
    }

}