import React, { useState } from 'react';
import { View, TextInput, TouchableOpacity, Text, StyleSheet, Alert } from 'react-native';
import axios from 'axios';
import { useRouter } from 'expo-router';

export default function HomeScreen() {
  const [comment, setComment] = useState('');
  const router = useRouter();

  const classifyComment = async () => {
    try {
      const response = await axios.post('http://192.168.8.104:8000/classify-comment', {
        text: comment,
      });
      router.push({ pathname: '/(tabs)/screens/ResultScreen', params: { sentiment: response.data.sentiment } });
    } catch (error) {
      Alert.alert('Erreur', 'Impossible de classer le commentaire');
    }
  };

  return (
    <View style={styles.container}>
      <Text style={styles.title}>Analyse de Sentiment</Text>
      <Text style={styles.subtitle}>Entrez un commentaire en arabe pour analyser son sentiment</Text>
      <TextInput
        style={styles.input}
        placeholder="Écrivez votre commentaire ici..."
        placeholderTextColor="#888"
        value={comment}
        onChangeText={setComment}
        multiline
      />
      <TouchableOpacity style={styles.button} onPress={classifyComment}>
        <Text style={styles.buttonText}>Analyser</Text>
      </TouchableOpacity>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    padding: 25,
    justifyContent: 'center',
    backgroundColor: '#f8f9fa', // Fond très clair
  },
  title: {
    fontSize: 32,
    fontWeight: 'bold',
    color: '#2c3e50', // Texte bleu foncé
    textAlign: 'center',
    marginBottom: 10,
  },
  subtitle: {
    fontSize: 16,
    color: '#7f8c8d', // Texte gris
    textAlign: 'center',
    marginBottom: 30,
  },
  input: {
    height: 150,
    borderColor: '#bdc3c7', // Bordure grise
    borderWidth: 1,
    borderRadius: 15, // Bordures arrondies
    marginBottom: 25,
    padding: 15,
    fontSize: 16,
    color: '#34495e', // Texte bleu foncé
    backgroundColor: '#fff', // Fond blanc
    textAlignVertical: 'top', // Aligner le texte en haut
    shadowColor: '#000', // Ombre
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.1,
    shadowRadius: 8,
    elevation: 5, // Ombre pour Android
  },
  button: {
    backgroundColor: '#3498db', // Fond bleu
    padding: 15,
    borderRadius: 15, // Bordures arrondies
    alignItems: 'center',
    shadowColor: '#000', // Ombre
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.2,
    shadowRadius: 8,
    elevation: 5, // Ombre pour Android
  },
  buttonText: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#fff', // Texte blanc
  },
});