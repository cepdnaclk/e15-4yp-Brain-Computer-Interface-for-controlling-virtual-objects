using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class LeftWall: MonoBehaviour {
    
    public Text scoreText;
    private void OnTriggerEnter(Collider other) {
        Vector3 initPosition = new Vector3(0f,1.91f,-8.88f);
        other.gameObject.transform.position = initPosition;
        print("left wall hitted.");
        ScorePanel.score_dict["Left"].score += 1;
    }
}
