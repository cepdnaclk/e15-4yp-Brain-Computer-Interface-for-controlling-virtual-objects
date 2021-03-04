using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class RightWall : MonoBehaviour {
    private void OnTriggerEnter(Collider other) {
        Vector3 initPosition = new Vector3(0f,1.91f,-8.88f);
        other.gameObject.transform.localPosition = initPosition;
        print("right wall is hitted.");
        ScorePanel.score_dict["Right"].score += 1;
    }
}
