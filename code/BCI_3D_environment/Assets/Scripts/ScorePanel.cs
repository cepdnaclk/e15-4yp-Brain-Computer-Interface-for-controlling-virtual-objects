using System.Collections.Generic;
using TMPro;
using UnityEngine;
using UnityEngine.UI;

public class ScorePanel : MonoBehaviour {
    
    public TMP_Text leftScoreText;
    public TMP_Text rightScoreText;
    public static IDictionary<string, Score> score_dict = new Dictionary<string, Score>();
    
    // Update is called once per frame
    private void Start(){
        score_dict.Add("Left",new Score(0,"00:00"));
        score_dict.Add("Right",new Score(0,"00:00"));
    }
    
    void Update() {
        leftScoreText.text = "Score : " + score_dict["Left"].score;
        rightScoreText.text = "Score : " + score_dict["Right"].score;
    }
    
}
