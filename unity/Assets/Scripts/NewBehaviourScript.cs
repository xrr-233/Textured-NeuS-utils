using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.IO;

public class NewBehaviourScript : MonoBehaviour
{
    public string file;
    public bool hasMeta;
    public bool hasColor;

    // Start is called before the first frame update
    void Start()
    {
        string path = string.Format("{0}/Resources/{1}.txt", Application.dataPath, file);
        StreamReader sr = File.OpenText(path);
        string str = null;
        Vector3 scale = new Vector3(0.05f, 0.05f, 0.05f);
        if (hasMeta)
        {
            str = sr.ReadLine();
            string[] pointCoord = str.Split(' ');
            int raysO = int.Parse(pointCoord[0]), raysV = int.Parse(pointCoord[1]), vertices = int.Parse(pointCoord[2]);
            int H = int.Parse(pointCoord[3]), W = int.Parse(pointCoord[4]);
            List<GameObject> points = new List<GameObject>();
            while ((str = sr.ReadLine()) != null)
            {
                pointCoord = str.Split(' ');
                GameObject point = GameObject.CreatePrimitive(PrimitiveType.Sphere);
                point.transform.position = new Vector3(float.Parse(pointCoord[0]), float.Parse(pointCoord[1]), float.Parse(pointCoord[2]));
                point.transform.localScale = scale;
                points.Add(point);
            }
            Debug.DrawLine(points[0].transform.position, points[raysO].transform.position, Color.red, 10000);
            Debug.DrawLine(points[0].transform.position, points[raysO + W - 1].transform.position, Color.red, 10000);
            Debug.DrawLine(points[0].transform.position, points[raysO + raysV - 1].transform.position, Color.red, 10000);
            Debug.DrawLine(points[0].transform.position, points[raysO + raysV - W].transform.position, Color.red, 10000);
        }
        else
        {
            while ((str = sr.ReadLine()) != null)
            {
                string[] pointCoord = str.Split(' ');
                GameObject point = GameObject.CreatePrimitive(PrimitiveType.Sphere);
                point.transform.position = new Vector3(float.Parse(pointCoord[0]), float.Parse(pointCoord[1]), float.Parse(pointCoord[2]));
                point.transform.localScale = scale;
                if (hasColor)
                {
                    Color color = new Color(float.Parse(pointCoord[3]), float.Parse(pointCoord[4]), float.Parse(pointCoord[5]));
                    Material[] materials = point.transform.GetComponent<MeshRenderer>().materials;
                    materials[0].color = color;
                    point.transform.GetComponent<MeshRenderer>().materials = materials;
                }
            }
        }

        Matrix4x4 extrinsic = new Matrix4x4();
        extrinsic[0, 0] = 0.6502f;
        extrinsic[0, 1] = -0.4492f;
        extrinsic[0, 2] = 0.6127f;
        extrinsic[0, 3] = -0.7896f;
        extrinsic[1, 0] = 0.4382f;
        extrinsic[1, 1] = 0.8805f;
        extrinsic[1, 2] = 0.1806f;
        extrinsic[1, 3] = -0.2568f;
        extrinsic[2, 0] = -0.6207f;
        extrinsic[2, 1] = 0.1511f;
        extrinsic[2, 2] = 0.7694f;
        extrinsic[2, 3] = -1.1867f;
        extrinsic[3, 3] = 1f;
        Quaternion rotation = extrinsic.rotation;
        GameObject.Find("Sphere").transform.rotation = rotation;
    }

    // Update is called once per frame
    void Update()
    {
        
    }
}
